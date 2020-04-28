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

    xdim, zdim = 4, 2
    sigma_w = [np.sqrt(0.01), np.sqrt(0.01)]
    sigma_v = [np.sqrt(1), np.sqrt(1)]

    F = model.F_poly_trans(1, 1, T)
    H = model.H_only_pos_meas(1, 1)
    L = np.eye(xdim)
    M = np.eye(zdim)
    Q = model.Q_dd_poly_proc_noise(1, 1, T, sigma_w, 1)
    R = model.R_only_pos_meas_noise(1, sigma_v)

    # initial state and error convariance
    x = np.array([1, 0.2, 2, 0.3])

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
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, state_arr[0, :], linewidth=0.8)
    ax[0].plot(n, measure_arr[0, :], '.')
    ax[0].plot(n, prior_state_arr[0, :], linewidth=0.8)
    ax[0].plot(n, post_state_arr[0, :], linewidth=0.8)
    ax[0].legend(['real', 'measurement', 'prediction', 'estimation'])
    ax[0].set_title('x state')
    ax[1].plot(n, state_arr[2, :], linewidth=0.8)
    ax[1].plot(n, measure_arr[1, :], '.')
    ax[1].plot(n, prior_state_arr[2, :], linewidth=0.8)
    ax[1].plot(n, post_state_arr[2, :], linewidth=0.8)
    ax[1].legend(['real', 'measurement', 'prediction', 'estimation'])
    ax[1].set_title('y state')
    plt.show()

    print('mean of x innovation: {}'.format(innov_arr[0, :].mean()))
    print('mean of y innovation: {}'.format(innov_arr[1, :].mean()))
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, innov_arr[0, :], linewidth=0.8)
    ax[0].set_title('x innovation')
    ax[1].plot(n, innov_arr[1, :], linewidth=0.8)
    ax[1].set_title('y innovation')
    plt.show()

    print('Kalman gain:\n{}'.format(gain))

    # trajectory
    _, ax = plt.subplots()
    ax.scatter(state_arr[0, 0], state_arr[2, 0], s=120, c='r', marker='x')
    ax.plot(state_arr[0, :], state_arr[2, :], linewidth=0.8)
    ax.plot(measure_arr[0, :], measure_arr[1, :], linewidth=0.8)
    ax.plot(post_state_arr[0, :], post_state_arr[2, :], linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['real', 'measurement', 'estimation'])
    ax.set_title('trajectory')
    plt.show()


def ABGFilter_test():
    N, T = 200, 1

    xdim, zdim = 6, 2
    sigma_w = [np.sqrt(0.01), np.sqrt(0.01)]
    sigma_v = [np.sqrt(1), np.sqrt(1)]

    F = model.F_poly_trans(2, 1, T)
    H = model.H_only_pos_meas(2, 1)
    L = np.eye(xdim)
    M = np.eye(zdim)
    Q = model.Q_dd_poly_proc_noise(2, 1, T, sigma_w)
    R = model.R_only_pos_meas_noise(1, sigma_v)

    # initial state and error convariance
    x = np.array([1, 0.2, 0.1, 2, 0.3, 0.1])
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
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, state_arr[0, :], linewidth=0.8)
    ax[0].plot(n, measure_arr[0, :], '.')
    ax[0].plot(n, prior_state_arr[0, :], linewidth=0.8)
    ax[0].plot(n, post_state_arr[0, :], linewidth=0.8)
    ax[0].legend(['real', 'measurement', 'prediction', 'estimation'])
    ax[0].set_title('x state')
    ax[1].plot(n, state_arr[3, :], linewidth=0.8)
    ax[1].plot(n, measure_arr[1, :], '.')
    ax[1].plot(n, prior_state_arr[3, :], linewidth=0.8)
    ax[1].plot(n, post_state_arr[3, :], linewidth=0.8)
    ax[1].legend(['real', 'measurement', 'prediction', 'estimation'])
    ax[1].set_title('y state')
    plt.show()

    print('mean of x innovation: {}'.format(innov_arr[0, :].mean()))
    print('mean of y innovation: {}'.format(innov_arr[1, :].mean()))
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, innov_arr[0, :], linewidth=0.8)
    ax[0].set_title('x innovation')
    ax[1].plot(n, innov_arr[1, :], linewidth=0.8)
    ax[1].set_title('y innovation')
    plt.show()

    print('Kalman gain:\n{}'.format(gain))

    # trajectory
    _, ax = plt.subplots()
    ax.scatter(state_arr[0, 0], state_arr[3, 0], s=120, c='r', marker='x')
    ax.plot(state_arr[0, :], state_arr[3, :], linewidth=0.8)
    ax.plot(measure_arr[0, :], measure_arr[1, :], linewidth=0.8)
    ax.plot(post_state_arr[0, :], post_state_arr[3, :], linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['real', 'measurement', 'estimation'])
    ax.set_title('trajectory')
    plt.show()

if __name__ == '__main__':
    ABFilter_test()
    ABGFilter_test()