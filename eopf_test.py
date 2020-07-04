#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from tracklib.model.model import trajectory_cv, trajectory_generator
import numpy as np
from numpy.core.records import record
import scipy.linalg as lg
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


def plot_traj():
    record = {
        'interval': [0.1, 0.1],
        'start': [
            [0, 0, 0],
            [0, 100, 0],
        ],
        'pattern': [
            [
                {'model': 'cv', 'length': 333, 'velocity': [200, 0, 0]},
                {'model': 'ct', 'length': 333, 'turnrate': 10},
                {'model': 'ca', 'length': 333, 'acceleration': 3}
            ],
            [
                {'model': 'cv', 'length': 333, 'velocity': [0, 200, 0]},
                {'model': 'ct', 'length': 333, 'turnrate': 10},
                {'model': 'ca', 'length': 333, 'acceleration': 3}
            ]
        ],
        'noise': [100*np.eye(3), 100*np.eye(3)],
        'pd': [0.9, 0.9],
        'entries': 2
    }
    trajs_state, trajs_meas = trajectory_generator(record, seed=0)
    plt.plot(trajs_meas[0][0, :], trajs_meas[0][1, :])
    plt.plot(trajs_meas[1][0, :], trajs_meas[1][1, :])
    plt.axis('equal')
    plt.show()


def EOPFilter_test():
    np.random.seed(2020)
    N, T = 200, 5
    Ns, Neff = 300, 200
    df = 100
    tau = 20

    axis = 2
    xdim, zdim = 4, 2
    sigma_w = np.sqrt(0.01)
    sigma_v = [np.sqrt(1), np.sqrt(1)]

    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    Q = model.Q_cv_dd(1, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    # initial state and error convariance
    x = np.array([0, 250, 0, 250], dtype=float)

    eopf = ft.EOPFilter(F, H, Q, R, Ns, Neff, df=(axis + 1 + df * np.exp(-T / tau)))

    state_arr = np.empty((xdim, N))
    measure_arr = np.empty((zdim, N))
    prior_state_arr = np.empty((xdim, N))
    prior_cov_arr = np.empty((xdim, xdim, N))
    post_state_arr = np.empty((xdim, N))
    post_cov_arr = np.empty((xdim, xdim, N))
    prior_ext = []
    post_ext = []

    for n in range(-1, N):
        w = tlb.multi_normal(0, np.kron(np.eye(2), Q))
        v = tlb.multi_normal(0, R)

        x = F @ x + w
        z = H @ x + v
        if n == -1:
            x_init, P_init = init.cv_init(z, R, (300, 300))
            # x_init[1], x_init[3] = 250, 250
            eopf.init(x_init, P_init, 400 * np.eye(2))
            continue
        state_arr[:, n] = x
        measure_arr[:, n] = z

        eopf.predict()
        prior_state_arr[:, n] = eopf.state
        prior_cov_arr[:, :, n] = eopf.cov
        prior_ext.append(eopf.extension())

        # eopf.correct([z])
        # eopf.correct([z + [-20, 20], z, z + [20, -20]])
        eopf.correct([z + [-20, 20], z + [20, -20], z, z + [20, 20], z + [-20, -20]])
        post_state_arr[:, n] = eopf.state
        post_cov_arr[:, :, n] = eopf.cov
        post_ext.append(eopf.extension())
        print(n)

    print(eopf)

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
    for i in range(N):
        plot_ellipse(ax, post_state_arr[0, i], post_state_arr[2, i], post_ext[i], 200)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()


if __name__ == '__main__':
    # plot_traj()
    EOPFilter_test()