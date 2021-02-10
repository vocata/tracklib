#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tracklib as tlb
import tracklib.init as init
import tracklib.filter as ft
import tracklib.model as model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def EKFilter_test():
    N, T = 200, 1

    axis = 3
    xdim, zdim = 6, 3
    sigma_w = [np.sqrt(0.01), np.sqrt(0.01), np.sqrt(0.001)]
    sigma_v = [np.sqrt(0.1), np.sqrt(0.01), np.sqrt(0.01)]

    F = model.F_cv(axis, T)
    L = np.eye(xdim)
    Q = model.Q_cv_dd(axis, T, sigma_w)

    H = model.H_cv(axis)
    M = np.eye(zdim)
    R = model.R_cv(axis, sigma_v)

    x = np.array([0, 0.2, 0, 0.2, 0, 0.01], dtype=float)

    kf = ft.KFilter(F, L, H, M, Q, R)

    state_arr = np.empty((xdim, N))
    measure_arr = np.empty((zdim, N))
    prior_state_arr = np.empty((xdim, N))
    post_state_arr = np.empty((xdim, N))
    prior_cov_arr = np.empty((xdim, xdim, N))
    post_cov_arr = np.empty((xdim, xdim, N))

    f = lambda x, u: F @ x + u
    h = lambda x: np.array(tlb.cart2sph(*x[::2]), dtype=float)
    for n in range(-1, N):
        w = tlb.multi_normal(0, Q)
        v = tlb.multi_normal(0, R)

        x = f(x, 0) + L @ w
        z = h(x) + M @ v
        if n == -1:
            z_cart, R_cart = model.convert_meas(z, R, elev=True)
            x_init, P_init = init.cv_init(z_cart, R_cart, 1)
            kf.init(x_init, P_init)
            continue
        state_arr[:, n] = x
        measure_arr[:, n] = tlb.sph2cart(*z)

        kf.predict()
        prior_state_arr[:, n] = kf.state
        prior_cov_arr[:, :, n] = kf.cov

        z_cart, R_cart = model.convert_meas(z, R, elev=True)
        kf.correct(z_cart, R=R_cart)
        post_state_arr[:, n] = kf.state
        post_cov_arr[:, :, n] = kf.cov

    print(kf)

    state_err = state_arr - post_state_arr
    print('RMS: %s' % np.std(state_err, axis=1))

    # plot
    n = np.arange(N)
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(n, state_arr[0, :], linewidth=0.8)
    ax.plot(n, measure_arr[0, :], '.')
    ax.plot(n, prior_state_arr[0, :], linewidth=0.8)
    ax.plot(n, post_state_arr[0, :], linewidth=0.8)
    ax.legend(['real', 'meas', 'pred', 'esti'])
    ax.set_title('x state')
    ax = fig.add_subplot(312)
    ax.plot(n, state_arr[2, :], linewidth=0.8)
    ax.plot(n, measure_arr[1, :], '.')
    ax.plot(n, prior_state_arr[2, :], linewidth=0.8)
    ax.plot(n, post_state_arr[2, :], linewidth=0.8)
    ax.legend(['real', 'meas', 'pred', 'esti'])
    ax.set_title('y state')
    ax = fig.add_subplot(313)
    ax.plot(n, state_arr[4, :], linewidth=0.8)
    ax.plot(n, measure_arr[2, :], '.')
    ax.plot(n, prior_state_arr[4, :], linewidth=0.8)
    ax.plot(n, post_state_arr[4, :], linewidth=0.8)
    ax.legend(['real', 'meas', 'pred', 'esti'])
    ax.set_title('z state')
    plt.show(block=False)

    print('x prior error variance {}'.format(prior_cov_arr[0, 0, -1]))
    print('x posterior error variance {}'.format(post_cov_arr[0, 0, -1]))
    print('y prior error variance {}'.format(prior_cov_arr[2, 2, -1]))
    print('y posterior error variance {}'.format(post_cov_arr[2, 2, -1]))
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(n, prior_cov_arr[0, 0, :], linewidth=0.8)
    ax.plot(n, post_cov_arr[0, 0, :], linewidth=0.8)
    ax.legend(['pred', 'esti'])
    ax.set_title('x error variance/mean square error')
    ax = fig.add_subplot(312)
    ax.plot(n, prior_cov_arr[2, 2, :], linewidth=0.8)
    ax.plot(n, post_cov_arr[2, 2, :], linewidth=0.8)
    ax.legend(['pred', 'esti'])
    ax.set_title('y error variance/mean square error')
    ax = fig.add_subplot(313)
    ax.plot(n, prior_cov_arr[4, 4, :], linewidth=0.8)
    ax.plot(n, post_cov_arr[4, 4, :], linewidth=0.8)
    ax.legend(['pred', 'esti'])
    ax.set_title('z error variance/mean square error')
    plt.show(block=False)

    # trajectory
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(state_arr[0, 0], state_arr[2, 0], state_arr[4, 0], s=50, c='r', marker='x', label='start')
    ax.plot(state_arr[0, :], state_arr[2, :], state_arr[4, :], linewidth=0.8, label='real')
    ax.scatter(measure_arr[0, :], measure_arr[1, :], measure_arr[2, :], s=5, c='orange', label='meas')
    ax.plot(prior_state_arr[0, :], prior_state_arr[2, :], prior_state_arr[4, :], linewidth=0.8, label='prior esti')
    ax.plot(post_state_arr[0, :], post_state_arr[2, :], post_state_arr[4, :], linewidth=0.8, label='post esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()


if __name__ == '__main__':
    EKFilter_test()
