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


def KFilter_test():
    N, T = 200, 1

    axis = 2
    xdim, zdim = 4, 2
    sigma_w = [np.sqrt(0.01), np.sqrt(0.01)]
    sigma_v = [np.sqrt(1), np.sqrt(1)]

    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    L = np.eye(xdim)
    M = np.eye(zdim)
    Q = model.Q_cv_dd(axis, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    # initial state and error convariance
    x = np.array([1, 0.2, 2, 0.3], dtype=float)

    kf = ft.KFilter(F, L, H, M, Q, R, xdim, zdim)

    state_arr = np.empty((xdim, N))
    measure_arr = np.empty((zdim, N))
    prior_state_arr = np.empty((xdim, N))
    post_state_arr = np.empty((xdim, N))
    prior_cov_arr = np.empty((xdim, xdim, N))
    post_cov_arr = np.empty((xdim, xdim, N))
    innov_arr = np.empty((zdim, N))
    innov_cov_arr = np.empty((zdim, zdim, N))

    for n in range(-1, N):
        w = tlb.multi_normal(0, Q)
        v = tlb.multi_normal(0, R)

        x = F @ x + L @ w
        z = H @ x + M @ v
        if n == -1:
            x_init, P_init = init.single_point_init(z, R, 1)
            kf.init(x_init, P_init)
            continue
        state_arr[:, n] = x
        measure_arr[:, n] = z
        kf.step(z)

        prior_state, prior_cov = kf.prior_state, kf.prior_cov
        post_state, post_cov = kf.post_state, kf.post_cov
        innov, innov_cov = kf.innov, kf.innov_cov
        gain = kf.gain

        prior_state_arr[:, n] = prior_state
        post_state_arr[:, n] = post_state
        prior_cov_arr[:, :, n] = prior_cov
        post_cov_arr[:, :, n] = post_cov
        innov_arr[:, n] = innov
        innov_cov_arr[:, :, n] = innov_cov
    print(len(kf))
    print(kf)

    state_err = state_arr - post_state_arr
    print('RMS: %s' % np.std(state_err, axis=1))

    # plot
    n = np.arange(N)
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(n, state_arr[0, :], linewidth=0.8)
    ax.plot(n, measure_arr[0, :], '.')
    ax.plot(n, prior_state_arr[0, :], linewidth=0.8)
    ax.plot(n, post_state_arr[0, :], linewidth=0.8)
    ax.legend(['real', 'measurement', 'prediction', 'estimation'])
    ax.set_title('x state')
    ax = fig.add_subplot(212)
    ax.plot(n, state_arr[2, :], linewidth=0.8)
    ax.plot(n, measure_arr[1, :], '.')
    ax.plot(n, prior_state_arr[2, :], linewidth=0.8)
    ax.plot(n, post_state_arr[2, :], linewidth=0.8)
    ax.legend(['real', 'measurement', 'prediction', 'estimation'])
    ax.set_title('y state')
    plt.show()

    print('x prior error variance {}'.format(prior_cov_arr[0, 0, -1]))
    print('x posterior error variance {}'.format(post_cov_arr[0, 0, -1]))
    print('y prior error variance {}'.format(prior_cov_arr[2, 2, -1]))
    print('y posterior error variance {}'.format(post_cov_arr[2, 2, -1]))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(n, prior_cov_arr[0, 0, :], linewidth=0.8)
    ax.plot(n, post_cov_arr[0, 0, :], linewidth=0.8)
    ax.legend(['prediction', 'estimation'])
    ax.set_title('x error variance/mean square error')
    ax = fig.add_subplot(212)
    ax.plot(n, prior_cov_arr[2, 2, :], linewidth=0.8)
    ax.plot(n, post_cov_arr[2, 2, :], linewidth=0.8)
    ax.legend(['prediction', 'estimation'])
    ax.set_title('y error variance/mean square error')
    plt.show()

    print('mean of x innovation: {}'.format(innov_arr[0, :].mean()))
    print('mean of y innovation: {}'.format(innov_arr[1, :].mean()))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(n, innov_arr[0, :], linewidth=0.8)
    ax.set_title('x innovation')
    ax = fig.add_subplot(212)
    ax.plot(n, innov_arr[1, :], linewidth=0.8)
    ax.set_title('y innovation')
    plt.show()

    print('x innovation variance {}'.format(innov_cov_arr[0, 0, -1]))
    print('y innovation variance {}'.format(innov_cov_arr[1, 1, -1]))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(n, innov_cov_arr[0, 0, :], linewidth=0.8)
    ax.set_title('x innovation variance')
    ax = fig.add_subplot(212)
    ax.plot(n, innov_cov_arr[1, 1, :], linewidth=0.8)
    ax.set_title('y innovation variance')
    plt.show()

    print('Kalman gain:\n{}'.format(gain))

    # trajectory
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(state_arr[0, 0], state_arr[2, 0], s=50, c='r', marker='x', label='start')
    ax.plot(state_arr[0, :], state_arr[2, :], linewidth=0.8, label='real')
    ax.scatter(measure_arr[0, :], measure_arr[1, :], s=5, c='orange', label='meas')
    ax.plot(prior_state_arr[0, :], prior_state_arr[2, :], linewidth=0.8, label='prior esti')
    ax.plot(post_state_arr[0, :], post_state_arr[2, :], linewidth=0.8, label='post esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()

    # trajectory amination
    # from matplotlib.patches import Ellipse
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter(state_arr[0, 0], state_arr[2, 0], s=50, c='r', marker='x', label='start')
    # ax.plot(state_arr[0, :], state_arr[2, :], linewidth=0.8, label='real')
    # ax.scatter(measure_arr[0, :], measure_arr[1, :], s=5, c='orange', label='meas')
    # ax.plot(prior_state_arr[0, :], prior_state_arr[2, :], linewidth=0.8, label='prior esti')
    # ax.plot(post_state_arr[0, :], post_state_arr[2, :], linewidth=0.8, label='post esti')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.legend()
    # ax.set_title('trajectory')
    # plt.ion()
    # for i in range(N):
    #     prior_state = prior_state_arr[np.s_[::2], i]
    #     prior_cov = prior_cov_arr[np.s_[::2], np.s_[::2], i]
    #     post_state = post_state_arr[np.s_[::2], i]
    #     post_cov = post_cov_arr[np.s_[::2], np.s_[::2], i]
    #     origin = prior_state
    #     d, v = lg.eigh(prior_cov)
    #     width = 2 * np.sqrt(d[0])
    #     height = 2 * np.sqrt(d[1])
    #     angle = np.rad2deg(np.log(complex(v[0, 0], v[0, 1])).imag)
    #     e = Ellipse(origin, width, height, angle)
    #     e.set_facecolor('white')
    #     e.set_edgecolor('black')
    #     ax.add_patch(e)
    #     plt.pause(0.2)
    #     e.remove()
    #     origin = post_state
    #     d, v = lg.eigh(post_cov)
    #     width = 2 * np.sqrt(d[0])
    #     height = 2 * np.sqrt(d[1])
    #     angle = np.rad2deg(np.log(complex(v[0, 0], v[0, 1])).imag)
    #     e = Ellipse(origin, width, height, angle)
    #     e.set_facecolor('white')
    #     e.set_edgecolor('red')
    #     ax.add_patch(e)
    #     plt.pause(0.01)
    #     e.remove()
    # plt.ioff()
    # plt.show()


if __name__ == '__main__':
    KFilter_test()