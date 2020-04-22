#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
import tracklib as tlb
import tracklib.filter as ft
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

    x_dim, z_dim = 4, 2
    gsf = ft.MMFilter()
    models = []
    probs = []
    for i in range(model_n):
        # qx, qy = np.sqrt((i + 1) / 100), np.sqrt((i + 1) / 100)
        # rr, ra = np.sqrt((i + 1) / 10), np.sqrt((i + 1) / 100)
        qx, qy = np.sqrt(0.01), np.sqrt(0.01)
        rr, ra = np.sqrt(0.1), np.sqrt(0.01)

        F = model.F_poly_trans(1, 1, T)
        f = lambda x, u: F @ x
        L = np.eye(x_dim)
        Q = model.Q_dd_poly_proc_noise(1, 1, T, [qx, qy])

        h = lambda x: np.array([lg.norm(x[0: 2]), np.arctan2(x[1], x[0])])
        M = np.eye(z_dim)
        R = model.R_only_pos_meas_noise(1, [rr, ra])

        # sub_filter = ft.KFilter(F, L, H, M, Q, R)
        sub_filter = ft.EKFilterAN(f, L, h, M, Q, R, order=1, it=0)
        models.append(sub_filter)
        probs.append(1 / model_n)
    gsf.add_models(models, probs)

    # initial state and error convariance
    x = np.array([1, 2, 0.2, 0.3])
    P = []
    for i in range(model_n):
        P.append((10 + i * 5) * np.eye(x_dim))

    gsf.init(x, P)

    state_arr = np.empty((x_dim, N))
    measure_arr = np.empty((z_dim, N))
    weight_state_arr = np.empty((x_dim, N))
    prob_arr = np.empty((model_n, N))

    for n in range(N):
        w = tlb.multi_normal(0, Q)
        v = tlb.multi_normal(0, R)

        x = f(x, 0) + L @ w
        z = h(x) + M @ v
        state_arr[:, n] = x
        measure_arr[:, n] = tlb.pol2cart(z[0], z[1])
        gsf.step(z)
        
        weight_state_arr[:, n] = gsf.weighted_state
        prob_arr[:, n] = gsf.probs
    print(len(gsf))
    print(gsf)

    # plot
    n = np.arange(N)
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, state_arr[0, :], linewidth=0.8)
    ax[0].plot(n, measure_arr[0, :], '.')
    ax[0].plot(n, weight_state_arr[0, :], linewidth=0.8)
    ax[0].legend(['real', 'measurement', 'weighted esti'])
    ax[0].set_title('x state')
    ax[1].plot(n, state_arr[1, :], linewidth=0.8)
    ax[1].plot(n, measure_arr[1, :], '.')
    ax[1].plot(n, weight_state_arr[1, :], linewidth=0.8)
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
    ax.scatter(state_arr[0, 0], state_arr[1, 0], s=120, c='r', marker='x')
    ax.plot(state_arr[0, :], state_arr[1, :], linewidth=0.8)
    ax.plot(measure_arr[0, :], measure_arr[1, :], linewidth=0.8)
    ax.plot(weight_state_arr[0, :], weight_state_arr[1, :], linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['real', 'measurement', 'weighted esti'])
    ax.set_title('trajectory')
    plt.show()


if __name__ == '__main__':
    GSEKFilter_test()