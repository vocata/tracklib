#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
import tracklib as tlb
import tracklib.init as init
import tracklib.filter as ft
import tracklib.model as model
import matplotlib.pyplot as plt
'''
notes:
vector is preferably a column vector, otherwise
the program may yield uncertain result.
'''


def PFilter_test():
    N, T = 200, 1
    Ns = 300
    Neff = 100

    axis = 2
    xdim, zdim = 4, 2
    sigma_w = [np.sqrt(0.01), np.sqrt(0.01)]
    sigma_v = [np.sqrt(0.1), np.sqrt(0.01)]

    F = model.F_cv(axis, T)
    L = np.eye(xdim)
    f = lambda x, u: F @ x
    Q = model.Q_cv_dd(axis, T, sigma_w)

    M = np.eye(zdim)
    h = lambda x: np.array([lg.norm(x[::2]), np.arctan2(x[2], x[0])], dtype=float)
    R = model.R_cv(axis, sigma_v)

    x = np.array([1, 0.2, 2, 0.3], dtype=float)

    # pf = ft.SIRPFilter(f, L, h, M, Q, R, Ns=Ns, Neff=Neff, resample_alg='roulette')

    kernal = ft.EpanechnikovKernal(xdim, Ns)
    # kernal = ft.GuassianKernal(xdim, Ns)
    pf = ft.RPFilter(f, L, h, M, Q, R, Ns=Ns, Neff=Neff, kernal=kernal, resample_alg='roulette')

    state_arr = np.empty((xdim, N))
    measure_arr = np.empty((zdim, N))
    MMSE_arr = np.empty((xdim, N))

    for n in range(-1, N):
        w = tlb.multi_normal(0, Q)
        v = tlb.multi_normal(0, R)

        x = f(x, 0) + L @ w
        z = h(x) + M @ v
        if n == -1:
            x_init, P_init = init.single_point_init(z, R, 1)
            pf.init(x_init, P_init)
            continue
        state_arr[:, n] = x
        measure_arr[:, n] = tlb.pol2cart(z[0], z[1])
        pf.step(z) 

        MMSE = pf.MMSE
        MMSE_arr[:, n] = MMSE
    print(len(pf))
    print(pf)

    state_err = state_arr - MMSE_arr
    print('RMS: %s' % np.std(state_err, axis=1))

    # plot
    n = np.arange(N)
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(n, state_arr[0, :], linewidth=0.8)
    ax.plot(n, measure_arr[0, :], '.')
    ax.plot(n, MMSE_arr[0, :], linewidth=0.8)
    ax.legend(['real', 'measurement', 'MMSE'])
    ax.set_title('x state')
    ax = fig.add_subplot(212)
    ax.plot(n, state_arr[2, :], linewidth=0.8)
    ax.plot(n, measure_arr[1, :], '.')
    ax.plot(n, MMSE_arr[2, :], linewidth=0.8)
    ax.legend(['real', 'measurement', 'MMSE'])
    ax.set_title('y state')
    plt.show()

    # trajectory
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(state_arr[0, 0], state_arr[2, 0], s=50, c='r', marker='x', label='start')
    ax.plot(state_arr[0, :], state_arr[2, :], linewidth=0.8, label='real')
    ax.scatter(measure_arr[0, :], measure_arr[1, :], s=5, c='orange', label='meas')
    ax.plot(MMSE_arr[0, :], MMSE_arr[2, :], linewidth=0.8, label='MMSE')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()


if __name__ == '__main__':
    PFilter_test()
