#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tracklib as tlb
import tracklib.filter as ft
import tracklib.model as model
import matplotlib.pyplot as plt
'''
notes:
vector is preferably a column vector, otherwise
the program may yield uncertain result.
'''


def SeqKFilter_test():
    N, T = 200, 1

    xdim, zdim = 4, 2
    qx, qy = np.sqrt(0.01), np.sqrt(0.01)
    rx, ry = np.sqrt(1), np.sqrt(1)

    F = model.F_poly_trans(1, 1, T)
    H = model.H_only_pos_meas(1, 1)
    L = np.eye(xdim)
    M = np.eye(zdim)
    Q = model.Q_dd_poly_proc_noise(1, 1, T, [qx, qy])
    R = model.R_only_pos_meas_noise(1, [rx, ry])

    # initial state and error convariance
    x = np.array([1, 2, 0.2, 0.3])
    P = 100 * np.eye(xdim)

    seqkf = ft.SeqKFilter(F, L, H, M, Q, R, xdim, zdim)
    seqkf.init(x, P) 

    state_arr = np.zeros((xdim, N))
    measure_arr = np.zeros((zdim, N))
    prior_state_arr = np.zeros((xdim, N))
    post_state_arr = np.zeros((xdim, N))
    prior_cov_arr = np.zeros((xdim, xdim, N))
    post_cov_arr = np.zeros((xdim, xdim, N))

    for n in range(N):
        w = tlb.multi_normal(0, Q)
        v = tlb.multi_normal(0, R)

        x = F @ x + L @ w
        z = H @ x + M @ v
        state_arr[:, n] = x
        measure_arr[:, n] = z
        seqkf.step(z)

        prior_state, prior_cov = seqkf.prior_state, seqkf.prior_cov
        post_state, post_cov = seqkf.post_state, seqkf.post_cov

        prior_state_arr[:, n] = prior_state
        post_state_arr[:, n] = post_state
        prior_cov_arr[:, :, n] = prior_cov
        post_cov_arr[:, :, n] = post_cov
    print(len(seqkf))
    print(seqkf)

    state_err = state_arr - post_state_arr
    print('RMS: %s' % np.std(state_err, axis=1))

    # plot
    n = np.arange(N)
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, state_arr[0, :], linewidth=0.8)
    ax[0].plot(n, measure_arr[0, :], '.')
    ax[0].plot(n, prior_state_arr[0, :], linewidth=0.8)
    ax[0].plot(n, post_state_arr[0, :], linewidth=0.8)
    ax[0].legend(['real', 'measurement', 'prediction', 'estimation'])
    ax[0].set_title('x state')
    ax[1].plot(n, state_arr[1, :], linewidth=0.8)
    ax[1].plot(n, measure_arr[1, :], '.')
    ax[1].plot(n, prior_state_arr[1, :], linewidth=0.8)
    ax[1].plot(n, post_state_arr[1, :], linewidth=0.8)
    ax[1].legend(['real', 'measurement', 'prediction', 'estimation'])
    ax[1].set_title('y state')
    plt.show()

    print('x prior error variance {}'.format(prior_cov_arr[0, 0, -1]))
    print('x posterior error variance {}'.format(post_cov_arr[0, 0, -1]))
    print('y prior error variance {}'.format(prior_cov_arr[1, 1, -1]))
    print('y posterior error variance {}'.format(post_cov_arr[1, 1, -1]))
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, prior_cov_arr[0, 0, :], linewidth=0.8)
    ax[0].plot(n, post_cov_arr[0, 0, :], linewidth=0.8)
    ax[0].legend(['prediction', 'estimation'])
    ax[0].set_title('x error variance/mean square error')
    ax[1].plot(n, prior_cov_arr[1, 1, :], linewidth=0.8)
    ax[1].plot(n, post_cov_arr[1, 1, :], linewidth=0.8)
    ax[1].legend(['prediction', 'estimation'])
    ax[1].set_title('y error variance/mean square error')
    plt.show()

    # trajectory
    _, ax = plt.subplots()
    ax.scatter(state_arr[0, 0], state_arr[1, 0], s=120, c='r', marker='x')
    ax.plot(state_arr[0, :], state_arr[1, :], linewidth=0.8)
    ax.plot(measure_arr[0, :], measure_arr[1, :], linewidth=0.8)
    ax.plot(post_state_arr[0, :], post_state_arr[1, :], linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['real', 'measurement', 'estimation'])
    ax.set_title('trajectory')
    plt.show()


if __name__ == '__main__':
    SeqKFilter_test()