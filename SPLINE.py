'''
Functions for 1D and 2D spline Interpolation

Was not able to implement successfully within main or the compression function

'''
from bisect import bisect

import numpy as np

##################
# Functions for fwd 1D interpolation
# Was not able to successfully implement
#
# def coeff(u, n):
#     temp = u
#     for i in range(1,n):
#         temp = temp * (u-i)
#     return temp
#
# def factor(n):
#     f = 1
#     for i in range(2, n+1):
#         f *= i
#     return f
#
# def interp1d(data, i, j, k, val):
#     n = data.shape[0]
#     x = []
#     for a in range(n):
#         x.append(k)
#     for l in range(1, n):
#         for m in range(n-l):
#             data[m][l] = data[m+1][l-1]-data[m][l-1]
#     sum = data[0][0][0]
#     u = (val - x[0])/ (x[1]-x[0])
#     for l in range(1,n):
#         sum += (coeff(u,l)*data[0][l])/factor(l)
############################################

# y is the data at point i, j, k

class NatSpline:
    def __init__(self, data, y):
        self.data = data
        self.y = y
        self.n = data.shape[0] # or len(data)
        self.h = self.interval()
        self.a, self.b, self.c, self.d = self.find_coeff()
        self.m = self.natural_spline()

    def interval(self):
        return [(self.data[i + 1] - self.data[i]) for i in range(self.n - 1)]

    def natural_spline(self):
        m = []
        for i in range(self.n - 1):
            slope = (self.y[i + 1] - self.y[i]) / (self.data[i + 1] - self.data[i])
            m.append(slope)
        return m

    def find_coeff(self):
        a = self.y
        n = self.n
        h = self.h
        m = np.zeros((n, n))
        u = np.zeros((n, 1))
        m[0, 0] = m[n - 1, n - 1] = 1
        for i in range(1, n - 1):
            m[i, i - 1] = h[i - 1]
            m[i, i] = 2 * (h[i] + h[i - 1])
            m[i, i + 1] = h[i]
            u[i, 0] = 3 * ((a[i + 1] - a[i]) / h[i] - (a[i] - a[i - 1]) / h[i - 1])
        c = np.linalg.solve(m, u).flatten()
        b = [float((a[i + 1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3) for i in range(n - 1)]
        d = [float((c[i + 1] - c[i]) / 3 * h[i]) for i in range(n - 1)]
        return a, b, c, d

    def find_index(self, z):
        if z > self.data[self.n - 1] or z < self.data[0]:
            print('Out of range')
            return None
        return bisect(self.data, z) - 1

    def derivatives(self, z, i=None):
        if i is None:
            i = self.find_index(z)
        r = z - i
        dz = 3 * self.d[i] * r**2 + 2 * self.c[i] * r + self.b[i]
        ddz = 6 * self.d[i] * r + 2 * self.c[i]
        return dz, ddz

    def curvature(self, z):
        i = np.floor(z)
        if i < 0:
            i = 0
        elif i >= self.n:
            i = self.n - 1
        dz, ddz = self.derivatives(z, i)
        return ddz / ((1 + dz ** 2) ** 1.5)

    def interpolate(self, z):
        i = self.find_index(z)
        if i is None:
            return None
        r = z - self.data[i]
        return (self.d[i] * r**3) + (self.c[i] * r**2) + (self.b[i] * r) + (self.a[i])

    def linear(self, z):
        i = self.find_index(z)
        if i is None:
            return None
        r = z - self.data[i]
        if i < self.n - 1:
            return self.m[i] * r + self.y[i]
        elif i == self.n - 1:
            return self.m[i - 1] * r + self.y[i]


class Spline2D(NatSpline):
    def __init__(self, data, x, y):
        super().__init__(data, y)
        self.sx = NatSpline(data, x)
        self.sy = NatSpline(data, y)
        self.curv = [self.curvature(data) for data in self.data[0:self.n - 1]]

    def curvature(self, z):
            # self.curvature = super().curvature()
        dx, ddx = self.sx.derivatives(z)
        dy, ddy = self.sy.derivatives(z)
        return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)

    def _find_index(self, z):
        if z > self.data[self.n - 1] or z < self.data[0]:
            print('Out of range')
            return None
        return bisect(self.data, z) - 1

    def mode(self, z):
        i = self._find_index(z)
        if i is None:
            return None
        k1, sig1 = abs(np.mean(self.curv[0:i + 1])), np.std(self.curv[0:i + 1])
        k2, sig2 = abs(np.mean(self.curv[i::])), np.std(self.curv[i::])
        if k1 - 2 * sig1 < k2 < k1 + 2 * sig1:
            return 'curve'
        else:
            return 'linear'

    def interpolate(self, z):
        if z > self.data[self.n - 1] or z < self.data[0]:
            print('Out of range')
            return None
        mode = self.mode(z)
        if mode == 'curve':
            return self.sx.interpolate(z), self.sy.interpolate(z), 'curve'
        elif mode == 'linear':
            return self.sx.linear(z), self.sy.linear(z), 'linear'

