#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import tracklib.utils as utils
import matplotlib.pyplot as plt


def plot_ellipse(ax, x0, y0, C, N, *args, **kwargs):
    x, y = utils.ellip_point(x0, y0, C, N)
    ax.plot(x, y, *args, **kwargs)


def KochGTT_test():
    record = {
        'interval': [10] * 5,
        'start': [
            [0, 0, 0],
            [500 * np.cos(np.pi / 4), 500 * np.sin(np.pi / 4), 0],
            [1000 * np.cos(np.pi / 4), 1000 * np.sin(np.pi / 4), 0],
            [1500 * np.cos(np.pi / 4), 1500 * np.sin(np.pi / 4), 0],
            [2000 * np.cos(np.pi / 4), 2000 * np.sin(np.pi / 4), 0],
        ],
        'pattern': [
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+2000*np.pi/80},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+2000*np.pi/40},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+2000*np.pi/40},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': -10/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+1500*np.pi/80},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+1500*np.pi/40},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+1500*np.pi/40},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': -5/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+1000*np.pi/80},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+1000*np.pi/40},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+1000*np.pi/40},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 0/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+500*np.pi/80},  # v'=v+w*ΔR
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+500*np.pi/40},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+500*np.pi/40},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 5/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 10/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
        ],
        'noise': [model.R_cv(3, [500., 100., 0.])] * 5,
        'entries': 5
    }
    trajs_state, trajs_meas = model.trajectory_generator(record)
    trajs_meas = model.trajectory_with_pd(trajs_meas, pd=0.8)

    N = trajs_state[0].shape[0]
    T = 10
    tau = 4 * T
    entries = 5
    df = 60

    axis = 2
    zdim, xdim = 2, 4
    sigma_w = 0.1
    sigma_v = [500., 100.]

    F = model.F_cv(1, T)
    H = model.H_cv(1)
    Q = model.Q_cv_dd(1, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    eof = ft.KochEOFilter(F, H, Q, T, tau)

    prior_state_arr = np.empty((N, xdim))
    prior_cov_arr = np.empty((N, xdim, xdim))
    post_state_arr = np.empty((N, xdim))
    post_cov_arr = np.empty((N, xdim, xdim))
    prior_ext_arr = np.empty((N, zdim, zdim))
    post_ext_arr = np.empty((N, zdim, zdim))

    # remove the invalid measurements
    zs = []
    for i in range(N):
        z = [
            trajs_meas[j][i, :-1] for j in range(entries)
            if not np.any(np.isnan(trajs_meas[j][i, :-1]))
        ]
        zs.append(z)

    for n in range(N):
        if n == 0:
            z_mean = np.mean(zs[n], axis=0)
            ellip = 1000**2 * np.eye(2)
            x_init, P_init = init.cv_init(z_mean, R, (30, 30))
            P_init = P_init[:2, :2]
            x_init[1], x_init[3] = 200, -200
            eof.init(x_init, P_init, df, ellip)

            prior_state_arr[n, :] = eof.state
            prior_cov_arr[n, :, :] = eof.cov
            prior_ext_arr[n, :, :, ] = eof.extension

            post_state_arr[n, :] = eof.state
            post_cov_arr[n, :, :] = eof.cov
            post_ext_arr[n, :, :] = eof.extension
            continue

        eof.predict()
        prior_state_arr[n, :] = eof.state
        prior_cov_arr[n, :, :] = eof.cov
        prior_ext_arr[n, :, :, ] = eof.extension

        eof.correct(zs[n])
        post_state_arr[n, :] = eof.state
        post_cov_arr[n, :, :] = eof.cov
        post_ext_arr[n, :, :] = eof.extension
        print(n)

    print(eof)

    # plot
    n = np.arange(N)

    print('x prior error variance {}'.format(prior_cov_arr[-1, 0, 0]))
    print('x posterior error variance {}'.format(post_cov_arr[-1, 0, 0]))
    print('y prior error variance {}'.format(prior_cov_arr[-1, 2, 2]))
    print('y posterior error variance {}'.format(post_cov_arr[-1, 2, 2]))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(n, prior_cov_arr[:, 0, 0], linewidth=0.8)
    ax.plot(n, post_cov_arr[:, 0, 0], linewidth=0.8)
    ax.legend(['pred', 'esti'])
    ax.set_title('x error variance/mean square error')
    ax = fig.add_subplot(212)
    ax.plot(n, prior_cov_arr[:, 2, 2], linewidth=0.8)
    ax.plot(n, post_cov_arr[:, 2, 2], linewidth=0.8)
    ax.legend(['pred', 'esti'])
    ax.set_title('y error variance/mean square error')
    plt.show()

    # trajectory
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(entries):
        ax.scatter(trajs_meas[i][:, 0], trajs_meas[i][:, 1], marker='^', facecolors=None, edgecolors='k', s=8)
    for i in range(N):
        plot_ellipse(ax, post_state_arr[i, 0], post_state_arr[i, 2], post_ext_arr[i], 200)
    ax.plot(post_state_arr[:, 0], post_state_arr[:, 2], linewidth=0.8, label='post esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()


def FeldmannGTT_test():
    record = {
        'interval': [10] * 5,
        'start': [
            [0, 0, 0],
            [500 * np.cos(np.pi / 4), 500 * np.sin(np.pi / 4), 0],
            [1000 * np.cos(np.pi / 4), 1000 * np.sin(np.pi / 4), 0],
            [1500 * np.cos(np.pi / 4), 1500 * np.sin(np.pi / 4), 0],
            [2000 * np.cos(np.pi / 4), 2000 * np.sin(np.pi / 4), 0],
        ],
        'pattern': [
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+2000*np.pi/80},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+2000*np.pi/40},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+2000*np.pi/40},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': -10/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+1500*np.pi/80},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+1500*np.pi/40},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+1500*np.pi/40},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': -5/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+1000*np.pi/80},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+1000*np.pi/40},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+1000*np.pi/40},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 0/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+500*np.pi/80},  # v'=v+w*ΔR
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+500*np.pi/40},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+500*np.pi/40},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 5/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 10/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
        ],
        'noise': [model.R_cv(3, [500., 100., 0.])] * 5,
        'entries': 5
    }
    trajs_state, trajs_meas = model.trajectory_generator(record)
    trajs_meas = model.trajectory_with_pd(trajs_meas, pd=0.8)

    N = trajs_state[0].shape[0]
    T = 10
    tau = 4 * T
    entries = 5
    df = 60

    axis = 2
    zdim, xdim = 2, 4
    sigma_w = 30
    sigma_v = [500., 100.]

    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    Q = model.Q_cv_dd(axis, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    eof = ft.FeldmannEOFilter(F, H, Q, R, T, tau)

    prior_state_arr = np.empty((N, xdim))
    prior_cov_arr = np.empty((N, xdim, xdim))
    post_state_arr = np.empty((N, xdim))
    post_cov_arr = np.empty((N, xdim, xdim))
    prior_ext_arr = np.empty((N, zdim, zdim))
    post_ext_arr = np.empty((N, zdim, zdim))

    # remove the invalid measurements
    zs = []
    for i in range(N):
        z = [
            trajs_meas[j][i, :-1] for j in range(entries)
            if not np.any(np.isnan(trajs_meas[j][i, :-1]))
        ]
        zs.append(z)

    for n in range(N):
        if n == 0:
            z_mean = np.mean(zs[n], axis=0)
            ellip = 1000**2 * np.eye(2)
            x_init, P_init = init.cv_init(z_mean, R, (30, 30))
            x_init[1], x_init[3] = 200, -200
            eof.init(x_init, P_init, df, ellip)

            prior_state_arr[n, :] = eof.state
            prior_cov_arr[n, :, :] = eof.cov
            prior_ext_arr[n, :, :] = eof.extension

            post_state_arr[n, :] = eof.state
            post_cov_arr[n, :, :] = eof.cov
            post_ext_arr[n, :, :] = eof.extension
            continue

        eof.predict()
        prior_state_arr[n, :] = eof.state
        prior_cov_arr[n, :, :] = eof.cov
        prior_ext_arr[n, :, :] = eof.extension

        eof.correct(zs[n])
        post_state_arr[n, :] = eof.state
        post_cov_arr[n, :, :] = eof.cov
        post_ext_arr[n, :, :] = eof.extension
        print(n)

    print(eof)

    # plot
    n = np.arange(N)

    print('x prior error variance {}'.format(prior_cov_arr[-1, 0, 0]))
    print('x posterior error variance {}'.format(post_cov_arr[-1, 0, 0]))
    print('y prior error variance {}'.format(prior_cov_arr[-1, 2, 2]))
    print('y posterior error variance {}'.format(post_cov_arr[-1, 2, 2]))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(n, prior_cov_arr[:, 0, 0], linewidth=0.8)
    ax.plot(n, post_cov_arr[:, 0, 0], linewidth=0.8)
    ax.legend(['pred', 'esti'])
    ax.set_title('x error variance/mean square error')
    ax = fig.add_subplot(212)
    ax.plot(n, prior_cov_arr[:, 2, 2], linewidth=0.8)
    ax.plot(n, post_cov_arr[:, 2, 2], linewidth=0.8)
    ax.legend(['pred', 'esti'])
    ax.set_title('y error variance/mean square error')
    plt.show()

    # trajectory
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(entries):
        ax.scatter(trajs_meas[i][:, 0], trajs_meas[i][:, 1], marker='^', facecolors=None, edgecolors='k', s=8)
    for i in range(N):
        plot_ellipse(ax, post_state_arr[i, 0], post_state_arr[i, 2], post_ext_arr[i], 200)
    ax.plot(post_state_arr[:, 0], post_state_arr[:, 2], linewidth=0.8, label='post esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()


def LanGTT_test():
    record = {
        'interval': [10] * 5,
        'start': [
            [0, 0, 0],
            [500 * np.cos(np.pi / 4), 500 * np.sin(np.pi / 4), 0],
            [1000 * np.cos(np.pi / 4), 1000 * np.sin(np.pi / 4), 0],
            [1500 * np.cos(np.pi / 4), 1500 * np.sin(np.pi / 4), 0],
            [2000 * np.cos(np.pi / 4), 2000 * np.sin(np.pi / 4), 0],
        ],
        'pattern': [
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+2000*np.pi/80},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+2000*np.pi/40},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+2000*np.pi/40},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': -10/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+1500*np.pi/80},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+1500*np.pi/40},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+1500*np.pi/40},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': -5/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+1000*np.pi/80},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+1000*np.pi/40},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+1000*np.pi/40},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 0/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+500*np.pi/80},  # v'=v+w*ΔR
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+500*np.pi/40},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20, 'velocity': 300+500*np.pi/40},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 5/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 20, 'velocity': [300 * np.sin(np.pi / 4), -300 * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 2, 'turnrate': 45/20},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20},
                {'model': 'cv', 'length': 15, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 90/20},
                {'model': 'cv', 'length': 20, 'velocity': 300},
                {'model': 'ct', 'length': 2, 'turnrate': 10/20},
                {'model': 'cv', 'length': 10, 'velocity': 300},
            ],
        ],
        'noise': [model.R_cv(3, [500., 100., 0.])] * 5,
        'entries': 5
    }
    trajs_state, trajs_meas = model.trajectory_generator(record)
    trajs_meas = model.trajectory_with_pd(trajs_meas, pd=0.8)

    N = trajs_state[0].shape[0]
    T = 10
    delta = 10
    entries = 5
    df = 60

    axis = 2
    zdim, xdim = 2, 4
    sigma_w = 0.1
    sigma_v = [500., 100.]

    F = model.F_cv(1, T)
    H = model.H_cv(1)
    Q = model.Q_cv_dd(1, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    eof = ft.LanEOFilter(F, H, Q, R, delta)

    prior_state_arr = np.empty((N, xdim))
    prior_cov_arr = np.empty((N, xdim, xdim))
    post_state_arr = np.empty((N, xdim))
    post_cov_arr = np.empty((N, xdim, xdim))
    prior_ext_arr = np.empty((N, zdim, zdim))
    post_ext_arr = np.empty((N, zdim, zdim))

    # remove the invalid measurements
    zs = []
    for i in range(N):
        z = [
            trajs_meas[j][i, :-1] for j in range(entries)
            if not np.any(np.isnan(trajs_meas[j][i, :-1]))
        ]
        zs.append(z)

    for n in range(N):
        if n == 0:
            z_mean = np.mean(zs[n], axis=0)
            ellip = 1000**2 * np.eye(2)
            x_init, P_init = init.cv_init(z_mean, R, (30, 30))
            P_init = P_init[:2, :2]
            x_init[1], x_init[3] = 200, -200
            eof.init(x_init, P_init, df, ellip)

            prior_state_arr[n, :] = eof.state
            prior_cov_arr[n, :, :] = eof.cov
            prior_ext_arr[n, :, :, ] = eof.extension

            post_state_arr[n, :] = eof.state
            post_cov_arr[n, :, :] = eof.cov
            post_ext_arr[n, :, :] = eof.extension
            continue

        eof.predict()
        prior_state_arr[n, :] = eof.state
        prior_cov_arr[n, :, :] = eof.cov
        prior_ext_arr[n, :, :, ] = eof.extension

        eof.correct(zs[n])
        post_state_arr[n, :] = eof.state
        post_cov_arr[n, :, :] = eof.cov
        post_ext_arr[n, :, :] = eof.extension
        print(n)

    print(eof)

    # plot
    n = np.arange(N)

    print('x prior error variance {}'.format(prior_cov_arr[-1, 0, 0]))
    print('x posterior error variance {}'.format(post_cov_arr[-1, 0, 0]))
    print('y prior error variance {}'.format(prior_cov_arr[-1, 2, 2]))
    print('y posterior error variance {}'.format(post_cov_arr[-1, 2, 2]))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(n, prior_cov_arr[:, 0, 0], linewidth=0.8)
    ax.plot(n, post_cov_arr[:, 0, 0], linewidth=0.8)
    ax.legend(['pred', 'esti'])
    ax.set_title('x error variance/mean square error')
    ax = fig.add_subplot(212)
    ax.plot(n, prior_cov_arr[:, 2, 2], linewidth=0.8)
    ax.plot(n, post_cov_arr[:, 2, 2], linewidth=0.8)
    ax.legend(['pred', 'esti'])
    ax.set_title('y error variance/mean square error')
    plt.show()

    # trajectory
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(entries):
        ax.scatter(trajs_meas[i][:, 0], trajs_meas[i][:, 1], marker='^', facecolors=None, edgecolors='k', s=8)
    for i in range(N):
        plot_ellipse(ax, post_state_arr[i, 0], post_state_arr[i, 2], post_ext_arr[i], 200)
    ax.plot(post_state_arr[:, 0], post_state_arr[:, 2], linewidth=0.8, label='post esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()


if __name__ == '__main__':
    KochGTT_test()
    FeldmannGTT_test()
    LanGTT_test()