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


def UKFilter_test():
    N, T = 200, 1

    axis = 2
    xdim, zdim = 4, 2
    sigma_w = [np.sqrt(0.01), np.sqrt(0.01)]
    sigma_v = [np.sqrt(0.1), np.sqrt(0.01)]

    F = model.F_cv(axis, T)
    L = np.eye(xdim)
    # f = lambda x, u: F @ x
    f = lambda x, u, w: F @ x + L @ w
    Q = model.Q_cv_dd(axis, T, sigma_w)

    M = np.eye(zdim)
    # h = lambda x: np.array([lg.norm(x[::2]), np.arctan2(x[2], x[0])], dtype=float)
    h = lambda x, v: np.array([lg.norm(x[::2]), np.arctan2(x[2], x[0])], dtype=float) + M @ v
    R = model.R_cv(axis, sigma_v)

    x = np.array([1, 0.2, 2, 0.3], dtype=float)

    # factory = ft.SimplexSigmaPoints()
    # factory = ft.SphericalSimplexSigmaPoints()
    # factory = ft.SymmetricSigmaPoints()       # CKF
    factory = ft.ScaledSigmaPoints()

    ukf = ft.UKFilterNAN(f, h, Q, R, xdim, zdim, factory=factory)

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

        # x = f(x, 0) + L @ w
        # z = h(x) + M @ v
        x = f(x, 0, w)
        z = h(x, v)
        if n == -1:
            x_init, P_init = init.single_point_init(z, R, 1)
            ukf.init(x_init, P_init)
            continue
        state_arr[:, n] = x
        measure_arr[:, n] = tlb.pol2cart(z[0], z[1])
        ukf.step(z)

        prior_state, prior_cov = ukf.prior_state, ukf.prior_cov
        post_state, post_cov = ukf.post_state, ukf.post_cov
        innov, innov_cov = ukf.innov, ukf.innov_cov
        gain = ukf.gain

        prior_state_arr[:, n] = prior_state
        post_state_arr[:, n] = post_state
        prior_cov_arr[:, :, n] = prior_cov
        post_cov_arr[:, :, n] = post_cov
        innov_arr[:, n] = innov
        innov_cov_arr[:, :, n] = innov_cov
    print(len(ukf))
    print(ukf)

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


if __name__ == '__main__':
    UKFilter_test()
