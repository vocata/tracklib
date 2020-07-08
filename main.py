#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
import scipy.stats as st
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import matplotlib.pyplot as plt
'''
notes:
vector is preferably a column vector, otherwise
the program may yield uncertain result.
'''


def plot_ellipse(ax, x0, y0, C, N, *args, **kwargs):
    C = (C + C.T) / 2
    U, s, V = lg.svd(C)
    D = (U + V) / 2
    S = np.diag(s)

    theta = np.linspace(0, 2 * np.pi, N)
    x = np.cos(theta) * np.sqrt(S[0, 0])
    y = np.sin(theta) * np.sqrt(S[1, 1])
    X = np.dot(D, np.vstack((x, y)))

    ax.plot(x0 + X[0, :], y0 + X[1, :], *args, **kwargs)


def EOPFilter_test():
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
        'pd': [1] * 5,
        'entries': 5
    }
    trajs_state, trajs_meas = model.trajectory_generator(record)

    N = trajs_state[0].shape[1]
    T = 10
    entries = 5
    Ns = 1000
    Neff = Ns // 6
    df, tau= 100, 20

    axis = 2
    xdim = 4
    sigma_w = 0.05
    sigma_v = [100., 100.]

    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    Q = model.Q_cv_dd(1, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    eopf = ft.EOPFilter(F, H, Q, R, Ns, Neff, df=(axis + 1 + df * np.exp(-T / tau)))

    prior_state_arr = np.empty((xdim, N))
    prior_cov_arr = np.empty((xdim, xdim, N))
    post_state_arr = np.empty((xdim, N))
    post_cov_arr = np.empty((xdim, xdim, N))
    prior_ext = []
    post_ext = []

    # remove the invalid measurements
    zs = []
    for i in range(N):
        z = [
            trajs_meas[j][:-1, i] for j in range(entries)
            if not np.any(np.isnan(trajs_meas[j][:-1, i]))
        ]
        zs.append(z)

    for n in range(N):
        if n == 0:
            z_mean = np.mean(zs[n], axis=0)
            ellip = 1000**2 * np.eye(2)
            x_init, P_init = init.cv_init(z_mean, R, (30, 30))
            x_init[1], x_init[3] = 200, 200
            eopf.init(x_init, P_init, 50, ellip)

            prior_state_arr[:, n] = eopf.state
            prior_cov_arr[:, :, n] = eopf.cov
            prior_ext.append(eopf.extension)

            post_state_arr[:, n] = eopf.state
            post_cov_arr[:, :, n] = eopf.cov
            post_ext.append(eopf.extension)
            continue

        eopf.predict()
        prior_state_arr[:, n] = eopf.state
        prior_cov_arr[:, :, n] = eopf.cov
        prior_ext.append(eopf.extension)

        eopf.correct(zs[n])
        post_state_arr[:, n] = eopf.state
        post_cov_arr[:, :, n] = eopf.cov
        post_ext.append(eopf.extension)
        print(n)

    print(eopf)

    # plot
    n = np.arange(N)

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
    for i in range(entries):
        ax.scatter(trajs_meas[i][0, :], trajs_meas[i][1, :], marker='^', facecolors=None, edgecolors='k', s=5)
    for i in range(N):
        plot_ellipse(ax, post_state_arr[0, i], post_state_arr[2, i], post_ext[i], 200)
    ax.plot(post_state_arr[0, :], post_state_arr[2, :], linewidth=0.8, label='post esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()


def IMMEOPFilter_test():
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
        'pd': [1] * 5,
        'entries': 5
    }
    trajs_state, trajs_meas = model.trajectory_generator(record)

    N = trajs_state[0].shape[1]
    T = 10
    entries = 5
    Ns = 1000
    Neff = Ns // 6

    axis = 2
    xdim = 4
    sigma_w = 0.05
    sigma_v = [100., 100.]

    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    Q = model.Q_cv_dd(1, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    theta1 = np.deg2rad(90 / 2)
    theta2 = np.deg2rad(45 / 2)
    A1 = np.array([[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1), np.cos(theta1)]])
    A2 = np.array([[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]])

    def size_fcn(mat):
        return st.wishart.rvs(40, mat / 40)
    def rotate_left1(mat):
        return st.wishart.rvs(70, A1 @ mat @ A1.T / 70)
    def rotate_left2(mat):
        return st.wishart.rvs(70, A2 @ mat @ A2.T / 70)

    trans_fcn = [size_fcn, rotate_left1, rotate_left2]
    immeopf = ft.IMMEOPFilter(F, H, Q, R, Ns, Neff, trans_fcn, trans_mat=0.9)

    prior_state_arr = np.empty((xdim, N))
    prior_cov_arr = np.empty((xdim, xdim, N))
    post_state_arr = np.empty((xdim, N))
    post_cov_arr = np.empty((xdim, xdim, N))
    prob_arr = np.empty((len(trans_fcn), N))
    prior_ext = []
    post_ext = []

    # remove the invalid measurements
    zs = []
    for i in range(N):
        z = [
            trajs_meas[j][:-1, i] for j in range(entries)
            if not np.any(np.isnan(trajs_meas[j][:-1, i]))
        ]
        zs.append(z)

    for n in range(N):
        if n == 0:
            z_mean = np.mean(zs[n], axis=0)
            ellip = 1000**2 * np.eye(2)
            x_init, P_init = init.cv_init(z_mean, R, (30, 30))
            x_init[1], x_init[3] = 200, 200
            immeopf.init(x_init, P_init, 50, ellip)

            prior_state_arr[:, n] = immeopf.state
            prior_cov_arr[:, :, n] = immeopf.cov
            prior_ext.append(immeopf.extension)

            post_state_arr[:, n] = immeopf.state
            post_cov_arr[:, :, n] = immeopf.cov
            post_ext.append(immeopf.extension)

            prob_arr[:, n] = immeopf.probs()
            continue

        immeopf.predict()
        prior_state_arr[:, n] = immeopf.state
        prior_cov_arr[:, :, n] = immeopf.cov
        prior_ext.append(immeopf.extension)

        immeopf.correct(zs[n])
        post_state_arr[:, n] = immeopf.state
        post_cov_arr[:, :, n] = immeopf.cov
        post_ext.append(immeopf.extension)

        prob_arr[:, n] = immeopf.probs()

        print(n, immeopf.probs())

    print(immeopf)

    # plot
    n = np.arange(N)

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
    for i in range(entries):
        ax.scatter(trajs_meas[i][0, :], trajs_meas[i][1, :], marker='^', facecolors=None, edgecolors='k', s=8)
    for i in range(N):
        plot_ellipse(ax, post_state_arr[0, i], post_state_arr[2, i], post_ext[i], 200)
    ax.plot(post_state_arr[0, :], post_state_arr[2, :], linewidth=0.8, label='post esti')
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
    model_types = ['line', '90/2', '45/2']
    for i in range(len(trans_fcn)):
        ax.plot(n, prob_arr[i, :], linewidth=0.8, label=model_types[i])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend()
    ax.set_title('models probability')
    plt.show()


if __name__ == '__main__':
    # EOPFilter_test()
    IMMEOPFilter_test()