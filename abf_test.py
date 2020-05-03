#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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


def ABFilter_test():
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

    alpha, beta = ft.get_alpha_beta(sigma_w, sigma_v, T)
    abf = ft.AlphaBetaFilter(alpha, beta, xdim, zdim, T)

    state_arr = np.empty((xdim, N))
    measure_arr = np.empty((zdim, N))
    prior_state_arr = np.empty((xdim, N))
    post_state_arr = np.empty((xdim, N))
    innov_arr = np.empty((zdim, N))

    for n in range(-1, N):
        w = tlb.multi_normal(0, Q)
        v = tlb.multi_normal(0, R)

        x = F @ x + L @ w
        z = H @ x + M @ v
        if n == -1:
            x_init, _ = init.single_point_init(z, R, 1)
            abf.init(x_init)
            continue
        state_arr[:, n] = x
        measure_arr[:, n] = z
        abf.step(z)

        prior_state = abf.prior_state
        post_state = abf.post_state
        innov = abf.innov
        gain = abf.gain

        prior_state_arr[:, n] = prior_state
        post_state_arr[:, n] = post_state
        innov_arr[:, n] = innov
    print(len(abf))
    print(abf)

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


def ABGFilter_test():
    N, T = 200, 1

    axis = 2
    xdim, zdim = 6, 2
    sigma_w = [np.sqrt(0.01), np.sqrt(0.01)]
    sigma_v = [np.sqrt(1), np.sqrt(1)]

    F = model.F_ca(axis, T)
    H = model.H_ca(axis)
    L = np.eye(xdim)
    M = np.eye(zdim)
    Q = model.Q_ca_dd(axis, T, sigma_w)
    R = model.R_ca(axis, sigma_v)

    # initial state and error convariance
    x = np.array([1, 0.2, 0.1, 2, 0.3, 0.1], dtype=float)
    x_init = x

    alpha, beta, gamma = ft.get_alpha_beta_gamma(sigma_w, sigma_v, T)
    abgf = ft.AlphaBetaGammaFilter(alpha, beta, gamma, xdim, zdim, T)

    state_arr = np.empty((xdim, N))
    measure_arr = np.empty((zdim, N))
    prior_state_arr = np.empty((xdim, N))
    post_state_arr = np.empty((xdim, N))
    innov_arr = np.empty((zdim, N))

    for n in range(-1, N):
        w = tlb.multi_normal(0, Q)
        v = tlb.multi_normal(0, R)

        x = F @ x + L @ w
        z = H @ x + M @ v
        if n == -1:
            abgf.init(x_init)
            continue
        state_arr[:, n] = x
        measure_arr[:, n] = z
        abgf.step(z)

        prior_state = abgf.prior_state
        post_state = abgf.post_state
        innov = abgf.innov
        gain = abgf.gain

        prior_state_arr[:, n] = prior_state
        post_state_arr[:, n] = post_state
        innov_arr[:, n] = innov
    print(len(abgf))
    print(abgf)

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
    ax.plot(n, state_arr[3, :], linewidth=0.8)
    ax.plot(n, measure_arr[1, :], '.')
    ax.plot(n, prior_state_arr[3, :], linewidth=0.8)
    ax.plot(n, post_state_arr[3, :], linewidth=0.8)
    ax.legend(['real', 'measurement', 'prediction', 'estimation'])
    ax.set_title('y state')
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

    print('Kalman gain:\n{}'.format(gain))

    # trajectory
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(state_arr[0, 0], state_arr[3, 0], s=50, c='r', marker='x', label='start')
    ax.plot(state_arr[0, :], state_arr[3, :], linewidth=0.8, label='real')
    ax.scatter(measure_arr[0, :], measure_arr[1, :], s=5, c='orange', label='meas')
    ax.plot(prior_state_arr[0, :], prior_state_arr[3, :], linewidth=0.8, label='prior esti')
    ax.plot(post_state_arr[0, :], post_state_arr[3, :], linewidth=0.8, label='post esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()

if __name__ == '__main__':
    ABFilter_test()
    ABGFilter_test()