#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
import scipy.stats as st
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import tracklib.utils as utils
import matplotlib.pyplot as plt


def gen_ellipse_uniform(trajs, C, R, theta, lamb):
    N = trajs.shape[1]
    trajs_elli = []
    elli = []
    A = np.eye(2)
    for i in range(N):
        pt_N = st.poisson.rvs(lamb)
        A = utils.rotate_matrix_deg(theta[i]) @ A
        elli.append(A @ C @ A.T)
        z = utils.ellipse_uniform(elli[i], pt_N)
        z += trajs[:, i]
        z += st.multivariate_normal.rvs(cov=R, size=pt_N)
        trajs_elli.append(z)
    return trajs_elli, elli

def EOT_test():
    vel = 50e3 / 36e2
    record = {
        'interval': [10],
        'start': [
            [0, 0, 0],
        ],
        'pattern': [
            [
                {'model': 'cv', 'length': 41, 'velocity': [vel * np.sin(np.pi / 4), -vel * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 9, 'turnrate': 45/90},
                {'model': 'cv', 'length': 27, 'velocity': vel},
                {'model': 'ct', 'length': 15, 'turnrate': 90/150},
                {'model': 'cv', 'length': 18, 'velocity': vel},
                {'model': 'ct', 'length': 15, 'turnrate': 90/150},
                {'model': 'cv', 'length': 54, 'velocity': vel},
            ],
        ],
        'noise': [model.R_cv(3, [0., 0., 0.])],
        'pd': [1],
        'entries': 1
    }
    trajs_state, trajs_meas = model.trajectory_generator(record)

    N = trajs_state[0].shape[1]
    T = 10
    entries = 1
    Ns = 2000
    Neff = Ns
    df = 40
    C = np.diag([340 / 2, 80 / 2])**2
    lamb = 20 / utils.ellipsoidal_volume(C)

    axis = 2
    xdim = 4
    sigma_w = 0.05
    sigma_v = [50, 50]

    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    Q = model.Q_cv_dd(1, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    theta = [-45]
    theta.extend([0] * 41)
    theta.extend([45 / 9 for i in range(9)])
    theta.extend([0] * 27)
    theta.extend([90 / 15 for i in range(15)])
    theta.extend([0] * 18)
    theta.extend([90 / 15 for i in range(15)])
    theta.extend([0] * 54)
    trajs_meas_elli, elli = gen_ellipse_uniform(trajs_meas[0][:-1], C, R, theta, 20)

    eopf = ft.EOPFilter(F, H, Q, R, Ns, Neff, df, lamb=lamb)

    prior_state_arr = np.empty((xdim, N))
    prior_cov_arr = np.empty((xdim, xdim, N))
    post_state_arr = np.empty((xdim, N))
    post_cov_arr = np.empty((xdim, xdim, N))
    prior_ext = []
    post_ext = []

    for n in range(N):
        z = trajs_meas_elli[n]
        if n == 0:
            z_mean = np.mean(z, axis=0)
            ellip = 100**2 * np.eye(2)
            x_init, P_init = init.cv_init(z_mean, R, (10, 10))
            x_init[1], x_init[3] = 14, -14
            eopf.init(x_init, P_init, df, ellip)

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

        eopf.correct(z)
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
        ax.scatter(trajs_meas[i][0, ::3], trajs_meas[i][1, ::3], marker='^', facecolors=None, edgecolors='k', s=5)
    for i in range(entries):
        for j in range(N):
            if j % 3 == 0:
                x, y = utils.ellipse_point(trajs_meas[i][0, j], trajs_meas[i][1, j], post_ext[j], 200)
                ax.plot(x, y, color='gray')
                x, y = utils.ellipse_point(trajs_meas[i][0, j], trajs_meas[i][1, j], elli[j], 200)
                ax.plot(x, y)

        for j in range(N):
            if j % 3 == 0:
                ax.scatter(trajs_meas_elli[j][:, 0], trajs_meas_elli[j][:, 1], s=1)
    ax.plot(post_state_arr[0, :], post_state_arr[2, :], linewidth=0.8, label='post esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()

if __name__ == '__main__':
    # test()
    EOT_test()