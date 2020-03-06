# -*- coding: utf-8 -*-

import numpy as np
from tracklib.utils import col

__all__ = ['lagrange_interp_poly', 'num_diff', 'num_diff2']


def lagrange_interp_poly(x, y=None):
    '''
    拉格朗日多项式
    '''
    x = col(x)
    N = len(x)

    if y is not None:
        y = col(y)

    Li = np.zeros((N, N))
    dLi = None
    d2Li = None
    if N == 1:
        Li = col(1)
    else:
        for cur_order in range(N):
            sel = [i for i in range(N) if i != cur_order]
            num_poly = np.poly(x[sel, 0])

            Li[:, cur_order] = num_poly / np.polyval(num_poly, x[cur_order])

        if N - 1 > 0:
            dLi = np.zeros((N - 1, N))
            for k in range(Li.shape[1]):
                dLi[:, k] = np.polyder(Li[:, k])
        if N - 2 > 0:
            d2Li = np.zeros((N - 2, N))
            for k in range(Li.shape[1]):
                d2Li[:, k] = np.polyder(dLi[:, k])

    if y is not None:
        Li = Li @ y
        dLi = None if dLi is None else dLi @ y
        d2Li = None if d2Li is None else d2Li @ y

    return Li, dLi, d2Li


# print(lagrange_interp_poly(3))
# print(lagrange_interp_poly(3, 2))
# print(lagrange_interp_poly([2, 5]))
# print(lagrange_interp_poly([2, 5], [1, 2]))
# print(lagrange_interp_poly([40, 66, 18, 71, 4, 28, 5, 10, 83, 70], [32, 96, 4, 44, 39, 77, 80, 19, 49, 45]))


def num_diff(x, f, f_dim, N=3, epsilon=None):
    '''
    一阶数值微分
    '''
    x = col(x)

    x_dim = x.shape[0]
    # If epsilon is not specified, then use some ad-hoc default value
    if epsilon is None:
        epsilon = col([max(eps, 1e-7) for eps in 1e-5 * np.abs(x)])
    else:
        epsilon = col([max(eps, 1e-7) for eps in epsilon])

    # 2*N+1 points fomula, error term is O(eps^(2*N))
    if N == 1:
        a, d = [1], 2
    elif N == 2:
        a, d = [8, -1], 12
    elif N == 3:
        a, d = [45, -9, 1], 60
    elif N == 4:
        a, d = [672, -168, 32, -3], 840
    elif N == 5:
        a, d = [2100, -600, 150, -25, 2], 2520
    elif N == 6:
        a, d = [23760, -7425, 2200, -495, 72, -5], 27720
    elif N == 7:
        a, d = [315315, -105105, 35035, -9555, 1911, -245, 15], 360360
    elif N == 8:
        a, d = [640640, -224224, 81536, -25480, 6272, -1120, 128, -7], 720720
    else:
        _, a, _ = lagrange_interp_poly(list(range(-N, N + 1)))
        a = (-a[-1, N - 1::-1]).tolist()
        d = 1

    p = len(a)
    J = np.zeros((f_dim, x_dim))
    for cur_el in range(x_dim):  # epsilon has same length as x
        eps = epsilon[cur_el]
        for cur_p in range(p):
            xp = x.copy()
            xp[cur_el] += (cur_p + 1) * eps  # partial derivation
            fxp = col(f(xp))
            J[:, cur_el] += a[cur_p] * fxp[:, 0]

            xp = x.copy()
            xp[cur_el] -= (cur_p + 1) * eps
            fxp = col(f(xp))
            J[:, cur_el] -= a[cur_p] * fxp[:, 0]
        J[:, cur_el] /= d * eps
    return J


# f = lambda x: [np.log(x), np.sin(x)]
# x = 0.25
# print(num_diff(x, f, 2))
# print(num_diff(x, f, 2, N=9))

# f = lambda x: [np.log(x[0])+np.sin(x[0]), np.log(x[1])+np.sin(x[1])]
# x = [1.0, 2.0]
# print(f(x))
# print(num_diff(x, f, 2))
# print(num_diff(x, f, 2, N=9))


def num_diff2(x, f, f_dim, N=1, epsilon=None):
    '''
    二阶数值微分
    '''
    x = col(x)

    x_dim = x.shape[0]
    # If epsilon is not specified, then use some ad-hoc default value
    if epsilon is None:
        epsilon = col([max(eps, 1e-7) for eps in 1e-5 * np.abs(x)])
    else:
        epsilon = col([max(eps, 1e-7) for eps in epsilon])

    # 2*N+1 points fomula, error term is O(eps^(2*N))
    if N == 1:
        a = [1, -2, 1]
    elif N == 2:
        a = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]
    elif N == 3:
        a = [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]
    else:
        _, _, a = lagrange_interp_poly(list(range(-N, N + 1)))
        a = a[-1, :].tolist()

    J2 = np.zeros((f_dim, x_dim))

    p = len(a)
    for cur_el in range(x_dim):  # epsilon has same length as x
        eps = epsilon[cur_el]
        for cur_p in range(p):
            xp = x.copy()
            xp[cur_el] += (cur_p - N) * eps
            fxp = col(f(xp))
            J2[:, cur_el] += a[cur_p] * fxp[:, 0]
        J2[:, cur_el] /= (eps**2)
    return J2


# f = lambda x: [np.log(x), np.sin(x)]
# x = 0.25
# print(num_diff2(x, f, 2))
# print(num_diff2(x, f, 2, N=5))

# f = lambda x: np.log(x)
# x = 1.0
# print(num_diff2(x, f, 1))
