import numpy as np
import sys
from dahuffman import HuffmanCodec

from scipy import interpolate
import zstd


def read_data(filename, dims):
    print(filename)
    data = np.fromfile(filename, dtype=np.float32)
    return data.reshape(dims)


def prediction(data, i, j, k):
    # Naive Prediction
    # return 0

    # 1D Lorenzo
    if k == 0:
        return 0
    return data[i, j, k - 1]

    ################################
    # 2D Lorenzo
    # if k == 0:
    #     return data[i, j, 0]
    # elif j == 0:
    #     return data[i, 0, k]
    # elif j == 0 & k == 0:
    #     return data[i, 0, 0]
    # else:
    #     return data[i, j, k - 1] + data[i, j-1, k] - data[i, j-1, k - 1]

    # 3D Lorenzo
    # if i == 0:
    #     return data[0, j, k]
    # elif j == 0:
    #     return data[i, 0, k]
    # elif k == 0:
    #     return data[i, j, 0]
    # if i == 0 & j == 0:
    #     return data[0, 0, k]
    # elif j == 0 & k == 0:
    #     return data[i, 0, 0]
    # elif i == 0 & k == 0:
    #     return data[0, j, 0]
    # if i == 0 & k == 0 & j == 0:
    #     return 0
    # else:
    #     return data[i - 1, j - 1, k - 1] + data[i, j - 1, k] + data[i - 1, j, k] + data[i, j, k - 1] - \
    #            data[i, j - 1, k - 1] - data[i - 1, j - 1, k] - data[i - 1, j, k - 1]


# Bicubic Interpolation
def quantize(diff, eb):
    index = int(np.abs(diff) / eb) + 1
    if index >= 65536:
        return -65536
    if index == 0:
        return 0
    index = index // 2
    if diff > 0:
        return index
    else:
        return - index


def dequantize(pred, eb, index):
    return pred + 2 * eb * index


def compress(data, dims, eb):
    # first compute original size, 4 is the bytes for  fp data, print the size
    # Initiliaze array quant integers (
    original_data_size = dims[0] * dims[1] * dims[2] * 4
    print("original data size = {}".format(original_data_size))
    quant_inds = np.zeros([dims[0] * dims[1] * dims[2]])
    quant_count = 0
    wild_data = []  # out of the range of compilated integer.
    # nested for loop to iterate through the dimensions.
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                # perform prediction
                pred = prediction(data, i, j, k)
                # perform quantization
                index = quantize(data[i, j, k] - pred, eb)
                if index == -65536:
                    wild_data.append(data[i, j, k])
                else:
                    # overwrite original data
                    data[i, j, k] = dequantize(pred, eb, index)
                # record the index
                quant_inds[quant_count] = index
                quant_count += 1
    wild_data_size = len(wild_data) * 4
    print("approximate wild data size = {}".format(wild_data_size))
    codec = HuffmanCodec.from_data(quant_inds)
    # codec.print_code_table()
    encoded = codec.encode(quant_inds)
    print("approximate quantization index size after Huffman = {}".format(len(encoded)))
    compressed = zstd.compress(encoded, 3)
    print("approximate quantization index size after ZSTD = {}".format(len(compressed)))
    compressed_size = wild_data_size + len(compressed)
    print("approximate compressed size = {}, compression ratio = {}".format(compressed_size,
                                                                            original_data_size * 1.0 / compressed_size))
    return codec, compressed, wild_data


def decompress(dims, eb, codec, compressed, wild_data):
    encoded = zstd.decompress(compressed)
    quant_inds = np.array(codec.decode(encoded))
    data = np.zeros([dims[0], dims[1], dims[2]], dtype=np.float32)
    wild_data_index = 0
    quant_count = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                # perform prediction
                pred = prediction(data, i, j, k)
                # get quant index
                index = quant_inds[quant_count]
                quant_count += 1
                # perform dequantization
                if index == -65536:
                    data[i, j, k] = wild_data[wild_data_index]
                    wild_data_index += 1
                else:
                    data[i, j, k] = dequantize(pred, eb, index)
    return data


def get_psnr(data, dec_data):
    data_range = np.max(data) - np.min(data)
    diff = data - dec_data
    rmse = np.sqrt(np.mean(diff ** 2))
    if rmse == 0:
        psnr = np.inf
    else:
        psnr = 20 * np.log10(data_range / rmse)
    return psnr


def evaluate(data, dec_data, abs_eb):
    ## used for debugging
    # for i in range(dims[0]):
    # 	for j in range(dims[1]):
    # 		for k in range(dims[2]):
    # 			if np.abs(data[i, j, k] - dec_data[i, j, k]) > abs_eb * 1.001:
    # 				print(i, j, k, data[i, j, k], dec_data[i, j, k])
    print("The required absolute error bound is {}".format(abs_eb))
    print("The maximal error in decompressed data is {}".format(np.max(np.abs(data - dec_data))))
    print("The PSNR of decompressed data is {}".format(get_psnr(data, dec_data)))


# read paramters
filename = sys.argv[1]
vrel_eb = float(sys.argv[2])  # relative error bound

dims = np.array([100, 500, 500])  # fixed
data = read_data(filename, dims)
abs_eb = (np.max(data) - np.min(data)) * vrel_eb  # absolute error bound, use 0.01
# reduce dims for testing
# dims = np.array([50, 50, 50])
data = data[:dims[0], :dims[1], :dims[2]]

print("start compression")
codec, compressed, wild_data = compress(data.copy(), dims, abs_eb)
# input is data (copy of data) need original for verification, param 2 is dimensions, 3 is abs error
# returns codec or the compressed bytes
# wild data, when it is too far from the value, have some data points you have to store '
# Can now compute decompressed data. 5 parameters
# dimensions, abs error, codec,
print("start decompression")
decompressed_data = decompress(dims, abs_eb, codec, compressed, wild_data)
print("evaluate errors")
evaluate(data, decompressed_data, abs_eb)
decompressed_data.tofile("decompressed.dat")
