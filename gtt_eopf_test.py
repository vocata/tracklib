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


def GTT_test():
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
    entries = 5
    T = 10
    Ns = 5000
    Neff = Ns // 3
    df = 50

    axis = 2
    zdim, xdim = 2, 4
    sigma_w = 0.05
    sigma_v = [500., 100.]

    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    Q = model.Q_cv_dd(1, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    eopf = ft.EOPFilter(F, H, Q, R, Ns, Neff, df)

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
            eopf.init(x_init, P_init, df, ellip)

            prior_state_arr[n, :] = eopf.state
            prior_cov_arr[n, :, :] = eopf.cov
            prior_ext_arr[n, :, :] = eopf.extension

            post_state_arr[n, :] = eopf.state
            post_cov_arr[n, :, :] = eopf.cov
            post_ext_arr[n, :, :] = eopf.extension
            continue

        eopf.predict()
        prior_state_arr[n, :] = eopf.state
        prior_cov_arr[n, :, :] = eopf.cov
        prior_ext_arr[n, :, :] = eopf.extension

        eopf.correct(zs[n])
        post_state_arr[n, :] = eopf.state
        post_cov_arr[n, :, :] = eopf.cov
        post_ext_arr[n, :, :] = eopf.extension
        print(n)

    print(eopf)

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


def GTT_RBP_test():
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
    entries = 5
    T = 10
    Ns = 2000
    Neff = Ns // 3
    df = 50

    axis = 2
    zdim, xdim = 2, 4
    sigma_w = 0.05
    sigma_v = [500., 100.]

    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    Q = model.Q_cv_dd(1, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    eopf = ft.EORBPFilter(F, H, Q, R, Ns, Neff, df)

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
            eopf.init(x_init, P_init, df, ellip)

            prior_state_arr[n, :] = eopf.state
            prior_cov_arr[n, :, :] = eopf.cov
            prior_ext_arr[n, :, :] = eopf.extension

            post_state_arr[n, :] = eopf.state
            post_cov_arr[n, :, :] = eopf.cov
            post_ext_arr[n, :, :] = eopf.extension
            continue

        eopf.predict()
        prior_state_arr[n, :] = eopf.state
        prior_cov_arr[n, :, :] = eopf.cov
        prior_ext_arr[n, :, :] = eopf.extension

        eopf.correct(zs[n])
        post_state_arr[n, :] = eopf.state
        post_cov_arr[n, :, :] = eopf.cov
        post_ext_arr[n, :, :] = eopf.extension
        print(n)

    print(eopf)

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


def GTT_RBP_TR_test():
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
    entries = 5
    T = 10
    Ns = 2000
    Neff = Ns // 3
    df = 50

    axis = 2
    zdim, xdim = 2, 4
    sigma_w = 0.05
    sigma_v = [500., 100.]

    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    Q = model.Q_cv_dd(1, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    eopf = ft.TurnRateEORBPFilter(F, H, Q, R, Ns, Neff, df, T, omega_std=1)

    prior_state_arr = np.empty((N, xdim))
    prior_cov_arr = np.empty((N, xdim, xdim))
    post_state_arr = np.empty((N, xdim))
    post_cov_arr = np.empty((N, xdim, xdim))
    prior_ext_arr = np.empty((N, zdim, zdim))
    post_ext_arr = np.empty((N, zdim, zdim))
    prior_omega_arr = np.empty((N, 1))
    post_omega_arr = np.empty((N, 1))

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
            eopf.init(x_init, P_init, df, ellip)

            prior_state_arr[n, :] = eopf.state
            prior_cov_arr[n, :, :] = eopf.cov
            prior_ext_arr[n, :, :] = eopf.extension
            prior_omega_arr[n, :] = eopf.omega()

            post_state_arr[n, :] = eopf.state
            post_cov_arr[n, :, :] = eopf.cov
            post_ext_arr[n, :, :] = eopf.extension
            post_omega_arr[n, :] = eopf.omega()
            continue

        eopf.predict()
        prior_state_arr[n, :] = eopf.state
        prior_cov_arr[n, :, :] = eopf.cov
        prior_ext_arr[n, :, :] = eopf.extension
        prior_omega_arr[n, :] = eopf.omega()

        eopf.correct(zs[n])
        post_state_arr[n, :] = eopf.state
        post_cov_arr[n, :, :] = eopf.cov
        post_ext_arr[n, :, :] = eopf.extension
        post_omega_arr[n, :] = eopf.omega()
        print(n)

    print(eopf)

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

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(n, prior_omega_arr, linewidth=0.8, label='prior omega')
    ax.plot(n, post_omega_arr, linewidth=0.8, label='post omega')
    ax.legend()
    ax.set_title('turning rate')

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
    GTT_test()
    GTT_RBP_test()
    GTT_RBP_TR_test()
