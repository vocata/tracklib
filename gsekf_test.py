#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
import tracklib as tlb
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import matplotlib.pyplot as plt
'''
notes:
vector is preferably a column vector, otherwise
the program may yield uncertain result.
'''


def GSEKFilter_test():
    '''
    Gaussian sum extended Kalman filter test program
    '''
    N, T = 200, 1
    model_n = 3

    xdim, zdim = 4, 2
    gsf = ft.MMFilter()
    models = []
    probs = []
    for i in range(model_n):
        sigma_w = [np.sqrt(0.01), np.sqrt(0.01)]
        sigma_v = [np.sqrt(0.1), np.sqrt(0.01)]

        F = model.F_poly_trans(1, 1, T)
        f = lambda x, u: F @ x
        L = np.eye(xdim)
        Q = model.Q_dd_poly_proc_noise(1, 1, T, sigma_w)

        h = lambda x: np.array([lg.norm(x[::2]), np.arctan2(x[2], x[0])], dtype=float)
        M = np.eye(zdim)
        R = model.R_only_pos_meas_noise(1, sigma_v)

        sub_filter = ft.EKFilterAN(f, L, h, M, Q, R, xdim, zdim, order=1, it=0)
        models.append(sub_filter)
        probs.append(1 / model_n)
    gsf.add_models(models, probs)

    # initial state and error convariance
    x = np.array([1, 0.2, 2, 0.3], dtype=float)
    P = []
    for i in range(model_n):
        P.append((10 + i * 5) * np.eye(xdim))
    gsf.init(x, P)

    state_arr = np.empty((xdim, N))
    measure_arr = np.empty((zdim, N))
    post_state_arr = np.empty((xdim, N))
    prob_arr = np.empty((model_n, N))

    for n in range(N):
        w = tlb.multi_normal(0, Q)
        v = tlb.multi_normal(0, R)

        x = f(x, 0) + L @ w
        z = h(x) + M @ v
        if n == -1:
            x_init, P_init = init.single_point_init(z, R, 1)
            gsf.init(x_init, P_init)
            continue
        state_arr[:, n] = x
        measure_arr[:, n] = tlb.pol2cart(z[0], z[1])
        gsf.step(z)
        
        post_state_arr[:, n] = gsf.post_state
        prob_arr[:, n] = gsf.probs()
    print(len(gsf))
    print(gsf)

    state_err = state_arr - post_state_arr
    print('RMS: %s' % np.std(state_err, axis=1))

    # plot
    n = np.arange(N)
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, state_arr[0, :], linewidth=0.8)
    ax[0].plot(n, measure_arr[0, :], '.')
    ax[0].plot(n, post_state_arr[0, :], linewidth=0.8)
    ax[0].legend(['real', 'measurement', 'weighted esti'])
    ax[0].set_title('x state')
    ax[1].plot(n, state_arr[2, :], linewidth=0.8)
    ax[1].plot(n, measure_arr[1, :], '.')
    ax[1].plot(n, post_state_arr[2, :], linewidth=0.8)
    ax[1].legend(['real', 'measurement', 'weighted esti'])
    ax[1].set_title('y state weighted estimation')
    plt.show()

    _, ax = plt.subplots()
    for i in range(model_n):
        ax.plot(n, prob_arr[i, :])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend([str(n) for n in range(model_n)])
    plt.show()

    # trajectory
    _, ax = plt.subplots()
    ax.scatter(state_arr[0, 0], state_arr[2, 0], s=120, c='r', marker='x')
    ax.plot(state_arr[0, :], state_arr[2, :], linewidth=0.8)
    ax.plot(measure_arr[0, :], measure_arr[1, :], linewidth=0.8)
    ax.plot(post_state_arr[0, :], post_state_arr[2, :], linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['real', 'measurement', 'weighted esti'])
    ax.set_title('trajectory')
    plt.show()


if __name__ == '__main__':
    GSEKFilter_test()
