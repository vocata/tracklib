#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import scipy.linalg as lg
import scipy.stats as st
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import tracklib.utils as utils
import matplotlib.pyplot as plt
'''
notes:
vector is preferably a column vector, otherwise
the program may yield uncertain result.
'''


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
        'pd': [0.8] * 5,
        'entries': 5
    }
    trajs_state, trajs_meas = model.trajectory_generator(record, seed=int(time.time()))

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
        'pd': [0.8] * 5,
        'entries': 5
    }
    trajs_state, trajs_meas = model.trajectory_generator(record, seed=int(time.time()))

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
        'pd': [0.8] * 5,
        'entries': 5
    }
    trajs_state, trajs_meas = model.trajectory_generator(record, seed=int(time.time()))

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


def IMMGTT_test1():
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
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+500*np.pi/80},
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
        'noise': [model.R_cv(3, [100., 100., 0.])] * 5,
        'pd': [0.8] * 5,
        'entries': 5
    }
    trajs_state, trajs_meas = model.trajectory_generator(record)

    N = trajs_state[0].shape[0]
    T = 10
    entries = 5
    Ns = 5000
    Neff = Ns // 3

    axis = 2
    zdim, xdim = 2, 4

    # cv1
    w_cv1 = 0.001
    v_cv1 = [100., 100.]
    df_cv1 = 100
    F_cv1 = model.F_cv(axis, T)
    H_cv1 = model.H_cv(axis)
    Q_cv1 = model.Q_cv_dd(1, T, w_cv1)
    R_cv1 = model.R_cv(axis, v_cv1)
    def cv1_state_trans_fcn(state, ext, cov, idx):
        mean = np.dot(F_cv1, state)
        cov = np.kron(ext, cov)
        return st.multivariate_normal.rvs(mean, cov)
    def cv1_ext_trans_fcn(ext, state, df, idx):
        return st.wishart.rvs(df, ext / df)
    def cv1_meas_fcn(state):
        return np.dot(H_cv1, state)
    # cv2
    w_cv2 = 0.01
    v_cv2 = [100., 100.]
    df_cv2 = 40
    F_cv2 = model.F_cv(axis, T)
    H_cv2 = model.H_cv(axis)
    Q_cv2 = model.Q_cv_dd(1, T, w_cv2)
    R_cv2 = model.R_cv(axis, v_cv2)
    theta = np.deg2rad(20)
    svar, cvar = np.sin(theta), np.cos(theta)
    A = np.array([[cvar, -svar], [svar, cvar]], dtype=float)
    def cv2_state_trans_fcn(state, ext, cov, idx):
        mean = np.dot(F_cv2, state)
        cov = np.kron(ext, cov)
        return st.multivariate_normal.rvs(mean, cov)
    def cv2_ext_trans_fcn(ext, state, df, idx):
        return st.wishart.rvs(df, A @ ext @ A.T / df)
    def cv2_meas_fcn(state):
        return np.dot(H_cv2, state)
    # cv3
    w_cv3 = 0.05
    v_cv3 = [100., 100.]
    df_cv3 = 40
    F_cv3 = model.F_cv(axis, T)
    H_cv3 = model.H_cv(axis)
    Q_cv3 = model.Q_cv_dd(1, T, w_cv3)
    R_cv3 = model.R_cv(axis, v_cv3)
    def cv3_state_trans_fcn(state, ext, cov, idx):
        mean = np.dot(F_cv3, state)
        cov = np.kron(ext, cov)
        return st.multivariate_normal.rvs(mean, cov)
    def cv3_ext_trans_fcn(ext, state, df, idx):
        return st.wishart.rvs(df, ext / df)
    def cv3_meas_fcn(state):
        return np.dot(H_cv3, state)

    state_trans_fcn = [cv1_state_trans_fcn, cv2_state_trans_fcn, cv3_state_trans_fcn]
    ext_trans_fcn = [cv1_ext_trans_fcn, cv2_ext_trans_fcn, cv3_ext_trans_fcn]
    meas_fcn = [cv1_meas_fcn, cv2_meas_fcn, cv3_meas_fcn]
    df = [df_cv1, df_cv2, df_cv3]
    state_noise = [Q_cv1, Q_cv2, Q_cv3]
    meas_noise = [R_cv1, R_cv2, R_cv3]

    # fcn definition
    def init_fcn(state, cov, df, extension, Ns):
        state_samples = st.multivariate_normal.rvs(state, cov, Ns)
        ext_samples = st.wishart.rvs(df, extension / df, Ns)
        return state_samples, ext_samples
    
    def merge_fcn(state_samples, ext_samples, weights, indices, Ns):
        ext = 0
        state = 0
        for i in range(Ns):
            ext += weights[i] * ext_samples[i]
            state += weights[i] * state_samples[i]
        cov = 0
        for i in range(Ns):
            err = state_samples[i] - state
            cov += weights[i] * np.outer(err, err)
        cov = (cov + cov.T) / 2
        return state, cov, ext

    immeopf = ft.IMMEOPFilter(len(df), init_fcn, state_trans_fcn, ext_trans_fcn, meas_fcn, merge_fcn, df, state_noise, meas_noise, Ns, Neff)

    prior_state_arr = np.empty((N, xdim))
    prior_cov_arr = np.empty((N, xdim, xdim))
    post_state_arr = np.empty((N, xdim))
    post_cov_arr = np.empty((N, xdim, xdim))
    prob_arr = np.empty((N, len(df)))
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
            x_init, P_init = init.cv_init(z_mean, R_cv1, (30, 30))
            df_init = 100
            x_init[1], x_init[3] = 200, -200
            immeopf.init(x_init, P_init, df_init, ellip)

            prior_state_arr[n, :] = immeopf.state
            prior_cov_arr[n, :, :] = immeopf.cov
            prior_ext_arr[n, :, :] = immeopf.extension

            post_state_arr[n, :] = immeopf.state
            post_cov_arr[n, :, :] = immeopf.cov
            post_ext_arr[n, :, :] = immeopf.extension

            prob_arr[n, :] = immeopf.probs()
            continue

        immeopf.predict()
        prior_state_arr[n, :] = immeopf.state
        prior_cov_arr[n, :, :] = immeopf.cov
        prior_ext_arr[n, :, :] = immeopf.extension

        immeopf.correct(zs[n])
        post_state_arr[n, :] = immeopf.state
        post_cov_arr[n, :, :] = immeopf.cov
        post_ext_arr[n, :, :] = immeopf.extension

        prob_arr[n, :] = immeopf.probs()

        print(n, immeopf.probs())

    print(immeopf)

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

    # model probability plot
    fig = plt.figure()
    ax = fig.add_subplot()
    n = np.arange(N)
    model_types = ['low', 'rotate', 'moderate']
    for i in range(len(df)):
        ax.plot(n, prob_arr[:, i], linewidth=0.8, label=model_types[i])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend()
    ax.set_title('models probability')
    plt.show()


def IMMGTT_test2():
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
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+500*np.pi/80},
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
        'noise': [model.R_cv(3, [100., 100., 0.])] * 5,
        'pd': [0.8] * 5,
        'entries': 5
    }
    trajs_state, trajs_meas = model.trajectory_generator(record)

    N = trajs_state[0].shape[0]
    T = 10
    entries = 5
    Ns = 5000
    Neff = Ns // 3

    axis = 2
    zdim, xdim = 2, 4

    # cv
    w_cv = 0.01
    v_cv = [100., 100.]
    df_cv = 60
    F_cv = lg.block_diag(model.F_cv(axis, T), 0)
    h_cv = model.h_ct(axis)
    Q_cv = model.Q_cv_dd(1, T, w_cv)
    R_cv = model.R_cv(axis, v_cv)
    def cv_state_trans_fcn(state, ext, cov, idx):
        mean = np.dot(F_cv, state)
        cov = lg.block_diag(np.kron(ext, cov), 0)
        return st.multivariate_normal.rvs(mean, cov)
    def cv_ext_trans_fcn(ext, state, df, idx):
        return st.wishart.rvs(df, ext / df)
    def cv_meas_fcn(state):
        return h_cv(state)
    # ct
    w_ct, w_omega = 0.01, 0.5
    v_ct = [100., 100.]
    df_ct = 100
    f_ct = model.f_ct(axis, T)
    h_ct = model.h_ct(axis)
    Q_ct = model.Q_cv_dd(1, T, w_ct)
    R_ct = model.R_cv(axis, v_ct)
    Q_w = (w_omega * T)**2
    def ct_state_trans_fcn(state, ext, cov, idx):
        mean = f_ct(state)
        if idx == 1:
            cov = lg.block_diag(np.kron(ext, cov), Q_w)
        else:
            cov = lg.block_diag(np.kron(ext, cov), 100)
        return st.multivariate_normal.rvs(mean, cov)
    def ct_ext_trans_fcn(ext, state, df, idx):
        theta = np.deg2rad(state[-1] * T)
        svar, cvar = np.sin(theta), np.cos(theta)
        A = np.array([[cvar, -svar], [svar, cvar]])
        return st.wishart.rvs(df, A @ ext @ A.T / df)
    def ct_meas_fcn(state):
        return h_ct(state)
    state_trans_fcn = [cv_state_trans_fcn, ct_state_trans_fcn]
    ext_trans_fcn = [cv_ext_trans_fcn, ct_ext_trans_fcn]
    meas_fcn = [cv_meas_fcn, ct_meas_fcn]
    df = [df_cv, df_ct]
    state_noise = [Q_cv, Q_ct]
    meas_noise = [R_cv, R_ct]

    # fcn definition
    def init_fcn(state, cov, df, extension, Ns):
        state_samples = np.zeros((Ns, state.size + 1))
        state_samples[:, :-1] = st.multivariate_normal.rvs(state, cov, Ns)
        ext_samples = st.wishart.rvs(df, extension / df, Ns)
        return state_samples, ext_samples
    
    def merge_fcn(state_samples, ext_samples, weights, indices, Ns):
        ext = 0
        state = 0
        for i in range(Ns):
            ext += weights[i] * ext_samples[i]
            state += weights[i] * state_samples[i]
        cov = 0
        for i in range(Ns):
            err = state_samples[i] - state
            cov += weights[i] * np.outer(err, err)
        cov = (cov + cov.T) / 2
        return state[:-1], cov[:-1, :-1], ext

    immeopf = ft.IMMEOPFilter(len(df), init_fcn, state_trans_fcn, ext_trans_fcn, meas_fcn, merge_fcn, df, state_noise, meas_noise, Ns, Neff)

    prior_state_arr = np.empty((N, xdim))
    prior_cov_arr = np.empty((N, xdim, xdim))
    post_state_arr = np.empty((N, xdim))
    post_cov_arr = np.empty((N, xdim, xdim))
    prob_arr = np.empty((N, len(df)))
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
            x_init, P_init = init.cv_init(z_mean, R_cv, (30, 30))
            df_init = 100
            x_init[1], x_init[3] = 200, -200
            immeopf.init(x_init, P_init, df_init, ellip)

            prior_state_arr[n, :] = immeopf.state
            prior_cov_arr[n, :, :] = immeopf.cov
            prior_ext_arr[n, :, :] = immeopf.extension

            post_state_arr[n, :] = immeopf.state
            post_cov_arr[n, :, :] = immeopf.cov
            post_ext_arr[n, :, :] = immeopf.extension

            prob_arr[n, :] = immeopf.probs()
            continue

        immeopf.predict()
        prior_state_arr[n, :] = immeopf.state
        prior_cov_arr[n, :, :] = immeopf.cov
        prior_ext_arr[n, :, :] = immeopf.extension

        immeopf.correct(zs[n])
        post_state_arr[n, :] = immeopf.state
        post_cov_arr[n, :, :] = immeopf.cov
        post_ext_arr[n, :, :] = immeopf.extension

        prob_arr[n, :] = immeopf.probs()

        print(n, immeopf.probs())

    print(immeopf)

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

    # model probability plot
    fig = plt.figure()
    ax = fig.add_subplot()
    n = np.arange(N)
    model_types = ['line', 'turn']
    for i in range(len(df)):
        ax.plot(n, prob_arr[:, i], linewidth=0.8, label=model_types[i])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend()
    ax.set_title('models probability')
    plt.show()


def IMMGTT_test3():
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
                {'model': 'ct', 'length': 2, 'turnrate': 45/20, 'velocity': 300+500*np.pi/80},
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
        'noise': [model.R_cv(3, [100., 100., 0.])] * 5,
        'pd': [0.8] * 5,
        'entries': 5
    }
    trajs_state, trajs_meas = model.trajectory_generator(record)

    N = trajs_state[0].shape[0]
    T = 10
    entries = 5
    Ns = 5000
    Neff = Ns // 3

    axis = 2
    zdim, xdim = 2, 4

    # cv1
    w_cv1 = 0.005
    v_cv1 = [100., 100.]
    df_cv1 = 100
    F_cv1 = model.F_cv(axis, T)
    H_cv1 = model.H_cv(axis)
    Q_cv1 = model.Q_cv_dd(1, T, w_cv1)
    R_cv1 = model.R_cv(axis, v_cv1)
    def cv1_state_trans_fcn(state, ext, cov, idx):
        mean = np.dot(F_cv1, state)
        cov = np.kron(ext, cov)
        return st.multivariate_normal.rvs(mean, cov)
    def cv1_ext_trans_fcn(ext, state, df, idx):
        return st.wishart.rvs(df, ext / df)
    def cv1_meas_fcn(state):
        return np.dot(H_cv1, state)
    # cv2
    w_cv2 = 0.1
    v_cv2 = [100., 100.]
    df_cv2 = 10
    F_cv2 = model.F_cv(axis, T)
    H_cv2 = model.H_cv(axis)
    Q_cv2 = model.Q_cv_dd(1, T, w_cv2)
    R_cv2 = model.R_cv(axis, v_cv2)
    def cv2_state_trans_fcn(state, ext, cov, idx):
        mean = np.dot(F_cv2, state)
        cov = np.kron(ext, cov)
        return st.multivariate_normal.rvs(mean, cov)
    def cv2_ext_trans_fcn(ext, state, df, idx):
        return st.wishart.rvs(df, ext / df)
    def cv2_meas_fcn(state):
        return np.dot(H_cv2, state)
    # cv3
    w_cv3 = 0.01
    v_cv3 = [100., 100.]
    df_cv3 = 40
    F_cv3 = model.F_cv(axis, T)
    H_cv3 = model.H_cv(axis)
    Q_cv3 = model.Q_cv_dd(1, T, w_cv3)
    R_cv3 = model.R_cv(axis, v_cv3)
    def cv3_state_trans_fcn(state, ext, cov, idx):
        mean = np.dot(F_cv3, state)
        cov = np.kron(ext, cov)
        return st.multivariate_normal.rvs(mean, cov)
    def cv3_ext_trans_fcn(ext, state, df, idx):
        return st.wishart.rvs(df, ext / df)
    def cv3_meas_fcn(state):
        return np.dot(H_cv3, state)

    state_trans_fcn = [cv1_state_trans_fcn, cv2_state_trans_fcn, cv3_state_trans_fcn]
    ext_trans_fcn = [cv1_ext_trans_fcn, cv2_ext_trans_fcn, cv3_ext_trans_fcn]
    meas_fcn = [cv1_meas_fcn, cv2_meas_fcn, cv3_meas_fcn]
    df = [df_cv1, df_cv2, df_cv3]
    state_noise = [Q_cv1, Q_cv2, Q_cv3]
    meas_noise = [R_cv1, R_cv2, R_cv3]

    # fcn definition
    def init_fcn(state, cov, df, extension, Ns):
        state_samples = st.multivariate_normal.rvs(state, cov, Ns)
        ext_samples = st.wishart.rvs(df, extension / df, Ns)
        return state_samples, ext_samples
    
    def merge_fcn(state_samples, ext_samples, weights, indices, Ns):
        ext = 0
        state = 0
        for i in range(Ns):
            ext += weights[i] * ext_samples[i]
            state += weights[i] * state_samples[i]
        cov = 0
        for i in range(Ns):
            err = state_samples[i] - state
            cov += weights[i] * np.outer(err, err)
        cov = (cov + cov.T) / 2
        return state, cov, ext

    immeopf = ft.IMMEOPFilter(len(df), init_fcn, state_trans_fcn, ext_trans_fcn, meas_fcn, merge_fcn, df, state_noise, meas_noise, Ns, Neff, trans_mat=0.95)

    prior_state_arr = np.empty((N, xdim))
    prior_cov_arr = np.empty((N, xdim, xdim))
    post_state_arr = np.empty((N, xdim))
    post_cov_arr = np.empty((N, xdim, xdim))
    prob_arr = np.empty((N, len(df)))
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
            x_init, P_init = init.cv_init(z_mean, R_cv1, (30, 30))
            df_init = 100
            x_init[1], x_init[3] = 200, -200
            immeopf.init(x_init, P_init, df_init, ellip)

            prior_state_arr[n, :] = immeopf.state
            prior_cov_arr[n, :, :] = immeopf.cov
            prior_ext_arr[n, :, :] = immeopf.extension

            post_state_arr[n, :] = immeopf.state
            post_cov_arr[n, :, :] = immeopf.cov
            post_ext_arr[n, :, :] = immeopf.extension

            prob_arr[n, :] = immeopf.probs()
            continue

        immeopf.predict()
        prior_state_arr[n, :] = immeopf.state
        prior_cov_arr[n, :, :] = immeopf.cov
        prior_ext_arr[n, :, :] = immeopf.extension

        immeopf.correct(zs[n])
        post_state_arr[n, :] = immeopf.state
        post_cov_arr[n, :, :] = immeopf.cov
        post_ext_arr[n, :, :] = immeopf.extension

        prob_arr[n, :] = immeopf.probs()

        print(n, immeopf.probs())

    print(immeopf)

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

    # model probability plot
    fig = plt.figure()
    ax = fig.add_subplot()
    n = np.arange(N)
    model_types = ['low', 'high', 'moderate']
    for i in range(len(df)):
        ax.plot(n, prob_arr[:, i], linewidth=0.8, label=model_types[i])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend()
    ax.set_title('models probability')
    plt.show()


if __name__ == '__main__':
    GTT_test()
    GTT_RBP_test()
    GTT_RBP_TR_test()
    # IMMGTT_test1()
    # IMMGTT_test2()
    # IMMGTT_test3()