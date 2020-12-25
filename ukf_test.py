#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
import tracklib as tlb
import tracklib.init as init
import tracklib.filter as ft
import tracklib.model as model
import matplotlib.pyplot as plt


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

    # pt_gen = ft.SimplexSigmaPoints()      # not suitable for non-additive noise
    # pt_gen = ft.SphericalSimplexSigmaPoints()
    # pt_gen = ft.SymmetricSigmaPoints()       # CKF
    pt_gen = ft.ScaledSigmaPoints()

    # ukf = ft.UKFilterAN(f, L, h, M, Q, R, point_generator=pt_gen)
    ukf = ft.UKFilterNAN(f, h, Q, R, point_generator=pt_gen)

    state_arr = np.empty((xdim, N))
    measure_arr = np.empty((zdim, N))
    prior_state_arr = np.empty((xdim, N))
    prior_cov_arr = np.empty((xdim, xdim, N))
    post_state_arr = np.empty((xdim, N))
    post_cov_arr = np.empty((xdim, xdim, N))

    for n in range(-1, N):
        w = tlb.multi_normal(0, Q)
        v = tlb.multi_normal(0, R)

        # x = f(x, 0) + L @ w
        # z = h(x) + M @ v
        x = f(x, 0, w)
        z = h(x, v)
        if n == -1:
            x_init, P_init = init.cv_init(z, R, 1)
            ukf.init(x_init, P_init)
            continue
        state_arr[:, n] = x
        measure_arr[:, n] = tlb.pol2cart(z[0], z[1])

        ukf.predict()
        prior_state_arr[:, n] = ukf.state
        prior_cov_arr[:, :, n] = ukf.cov

        ukf.correct(z)
        post_state_arr[:, n] = ukf.state
        post_cov_arr[:, :, n] = ukf.cov

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
    ax.legend(['real', 'meas', 'pred', 'esti'])
    ax.set_title('x state')
    ax = fig.add_subplot(212)
    ax.plot(n, state_arr[2, :], linewidth=0.8)
    ax.plot(n, measure_arr[1, :], '.')
    ax.plot(n, prior_state_arr[2, :], linewidth=0.8)
    ax.plot(n, post_state_arr[2, :], linewidth=0.8)
    ax.legend(['real', 'meas', 'pred', 'esti'])
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
    ax.legend(['pred', 'esti'])
    ax.set_title('x error variance/mean square error')
    ax = fig.add_subplot(212)
    ax.plot(n, prior_cov_arr[2, 2, :], linewidth=0.8)
    ax.plot(n, post_cov_arr[2, 2, :], linewidth=0.8)
    ax.legend(['pred', 'esti'])
    ax.set_title('y error variance/mean square error')
    plt.show()

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
