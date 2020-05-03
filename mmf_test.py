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


def MMFilter_test():
    N, T = 200, 1
    axis = 2
    xdim, zdim = 4, 2
    
    # model 1
    sigma_w1 = np.sqrt(10 * np.random.rand())
    sigma_v1 = np.sqrt(10 * np.random.rand())
    F1 = model.F_cv(axis, T)
    H1 = model.H_cv(axis)
    L1 = np.eye(xdim)
    M1 = np.eye(zdim)
    Q1 = model.Q_cv_dd(axis, T, sigma_w1)
    R1 = model.R_cv(axis, sigma_v1)
    cv_kf1 = ft.KFilter(F1, L1, H1, M1, Q1, R1, xdim, zdim)

    # model 2
    sigma_w2 = np.sqrt(10 * np.random.rand())
    sigma_v2 = np.sqrt(10 * np.random.rand())
    F2 = model.F_cv(axis, T)
    H2 = model.H_cv(axis)
    L2 = np.eye(xdim)
    M2 = np.eye(zdim)
    Q2 = model.Q_cv_dd(axis, T, sigma_w2)
    R2 = model.R_cv(axis, sigma_v2)
    cv_kf2 = ft.KFilter(F2, L2, H2, M2, Q2, R2, xdim, zdim)

    # model 3
    sigma_w3 = np.sqrt(10 * np.random.rand())
    sigma_v3 = np.sqrt(10 * np.random.rand())
    F3 = model.F_cv(axis, T)
    H3 = model.H_cv(axis)
    L3 = np.eye(xdim)
    M3 = np.eye(zdim)
    Q3 = model.Q_cv_dd(axis, T, sigma_w3)
    R3 = model.R_cv(axis, sigma_v3)
    cv_kf3 = ft.KFilter(F3, L3, H3, M3, Q3, R3, xdim, zdim)

    # initial state and error convariance
    x = np.array([1, 0.2, 2, 0.3], dtype=float)

    # number of models
    r = 3

    models = [cv_kf1, cv_kf2, cv_kf3]
    mmf = ft.MMFilter()
    mmf.add_models(models)

    state_arr = np.empty((xdim, N))
    measure_arr = np.empty((zdim, N))
    prior_state_arr = np.empty((xdim, N))
    post_state_arr = np.empty((xdim, N))
    prior_cov_arr = np.empty((xdim, xdim, N))
    post_cov_arr = np.empty((xdim, xdim, N))
    innov_arr = np.empty((zdim, N))
    innov_cov_arr = np.empty((zdim, zdim, N))
    prob_arr = np.empty((r, N + 1))

    prob_arr[:, 0] = mmf.probs()
    for n in range(-1, N):
        w = tlb.multi_normal(0, Q2)
        v = tlb.multi_normal(0, R2)

        x = F2 @ x + L2 @ w
        z = H2 @ x + M2 @ v
        if n == -1:
            x_init, P_init = init.single_point_init(z, R2, 1)
            mmf.init(x_init, P_init)
            continue
        state_arr[:, n] = x
        measure_arr[:, n] = z
        mmf.step(z)

        prior_state, prior_cov = mmf.prior_state, mmf.prior_cov
        post_state, post_cov = mmf.post_state, mmf.post_cov
        innov, innov_cov = mmf.innov, mmf.innov_cov

        prior_state_arr[:, n] = prior_state
        post_state_arr[:, n] = post_state
        prior_cov_arr[:, :, n] = prior_cov
        post_cov_arr[:, :, n] = post_cov
        innov_arr[:, n] = innov
        innov_cov_arr[:, :, n] = innov_cov
        prob_arr[:, n + 1] = mmf.probs()
    print(len(mmf))
    print(mmf)

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

    fig = plt.figure()
    ax = fig.add_subplot()
    n = np.arange(N + 1)
    types = ['cv1', 'cv2', 'cv3']
    for i in range(r):
        ax.plot(n, prob_arr[i, :], linewidth=0.8, label=types[i])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    MMFilter_test()