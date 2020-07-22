#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import tracklib.utils as utils
import matplotlib.pyplot as plt


def gen_ellipse_uniform(trajs, C, R, theta, lamb):
    N = trajs.shape[0]
    trajs_ellip = []
    real_ellip = []
    A = np.eye(2)
    for i in range(N):
        pt_N = st.poisson.rvs(lamb)
        A = utils.rotate_matrix_deg(theta[i]) @ A
        real_ellip.append(A @ C @ A.T)
        z = utils.ellip_uniform(real_ellip[i], pt_N)
        z += st.multivariate_normal.rvs(cov=R, size=pt_N)
        z += trajs[i]
        trajs_ellip.append(z)
    return trajs_ellip, real_ellip


def KochEOT_test():
    vel = 50e3 / 36e2   # 27 knot
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

    N = trajs_state[0].shape[0]
    T = 10
    tau = 8 * T
    entries = 1
    df = 50
    C = np.diag([340 / 2, 80 / 2])**2

    axis = 2
    zdim, xdim = 2, 4
    sigma_w = 0.05
    sigma_v = [50, 50]

    # below parameters are in single dimension
    F = model.F_cv(1, T)
    H = model.H_cv(1)
    Q = model.Q_cv_dd(1, T, sigma_w)
    # useless for Koch approach
    R = model.R_cv(axis, sigma_v)

    theta = [-45]
    theta.extend([0] * 41)
    theta.extend([45 / 9 for i in range(9)])
    theta.extend([0] * 27)
    theta.extend([90 / 15 for i in range(15)])
    theta.extend([0] * 18)
    theta.extend([90 / 15 for i in range(15)])
    theta.extend([0] * 54)
    trajs_meas_ellip, real_ellip = gen_ellipse_uniform(trajs_meas[0][:, :-1], C, R, theta, 20)

    epf = ft.KochEOFilter(F, H, Q, T, tau)

    prior_state_arr = np.empty((N, xdim))
    prior_cov_arr = np.empty((N, xdim, xdim))
    post_state_arr = np.empty((N, xdim))
    post_cov_arr = np.empty((N, xdim, xdim))
    prior_ext_arr = np.empty((N, zdim, zdim))
    post_ext_arr = np.empty((N, zdim, zdim))

    for n in range(N):
        zs = trajs_meas_ellip[n]
        if n == 0:
            z_mean = np.mean(zs, axis=0)
            ellip = 100**2 * np.eye(2)
            x_init, _ = init.cv_init(z_mean, R, (10, 10))
            P_init = np.diag([1., 1.])
            x_init[1], x_init[3] = 14, -14
            epf.init(x_init, P_init, df, ellip)

            prior_state_arr[n, :] = epf.state
            prior_cov_arr[n, :, :] = epf.cov
            prior_ext_arr[n, :, :] = epf.extension

            post_state_arr[n, :] = epf.state
            post_cov_arr[n, :, :] = epf.cov
            post_ext_arr[n, :, :] = epf.extension
            continue

        epf.predict()
        prior_state_arr[n, :] = epf.state
        prior_cov_arr[n, :, :] = epf.cov
        prior_ext_arr[n, :, :] = epf.extension

        epf.correct(zs)
        post_state_arr[n, :] = epf.state
        post_cov_arr[n, :, :] = epf.cov
        post_ext_arr[n, :, :] = epf.extension
        print(n)

    print(epf)

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
        ax.scatter(trajs_meas[i][::3, 0], trajs_meas[i][::3, 1], marker='^', facecolors=None, edgecolors='k', s=5)
    for i in range(entries):
        for j in range(N):
            if j % 3 == 0:
                x, y = utils.ellip_point(trajs_meas[i][j, 0], trajs_meas[i][j, 1], post_ext_arr[j], 200)
                ax.plot(x, y, color='gray')
                x, y = utils.ellip_point(trajs_meas[i][j, 0], trajs_meas[i][j, 1], real_ellip[j], 200)
                ax.plot(x, y, color='green')
                ax.scatter(trajs_meas_ellip[j][:, 0], trajs_meas_ellip[j][:, 1], s=1)
    ax.plot(post_state_arr[:, 0], post_state_arr[:, 2], linewidth=0.8, label='post esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()


def FeldmannEOT_test():
    vel = 50e3 / 36e2   # 27 knot
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

    N = trajs_state[0].shape[0]
    T = 10
    tau = 8 * T
    entries = 1
    df = 50
    C = np.diag([340 / 2, 80 / 2])**2

    axis = 2
    zdim, xdim = 2, 4
    sigma_w = 0.05
    sigma_v = [50, 50]

    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    Q = model.Q_cv_dd(1, T, sigma_w)    # single dimension process noise cov
    R = model.R_cv(axis, sigma_v)

    theta = [-45]
    theta.extend([0] * 41)
    theta.extend([45 / 9 for i in range(9)])
    theta.extend([0] * 27)
    theta.extend([90 / 15 for i in range(15)])
    theta.extend([0] * 18)
    theta.extend([90 / 15 for i in range(15)])
    theta.extend([0] * 54)
    trajs_meas_ellip, real_ellip = gen_ellipse_uniform(trajs_meas[0][:, :-1], C, R, theta, 20)

    epf = ft.FeldmannEOFilter(F, H, Q, R, T, tau)

    prior_state_arr = np.empty((N, xdim))
    prior_cov_arr = np.empty((N, xdim, xdim))
    post_state_arr = np.empty((N, xdim))
    post_cov_arr = np.empty((N, xdim, xdim))
    prior_ext_arr = np.empty((N, zdim, zdim))
    post_ext_arr = np.empty((N, zdim, zdim))

    for n in range(N):
        zs = trajs_meas_ellip[n]
        if n == 0:
            z_mean = np.mean(zs, axis=0)
            ellip = 100**2 * np.eye(2)
            x_init, P_init = init.cv_init(z_mean, R, (10, 10))
            x_init[1], x_init[3] = 14, -14
            epf.init(x_init, P_init, df, ellip)

            prior_state_arr[n, :] = epf.state
            prior_cov_arr[n, :, :] = epf.cov
            prior_ext_arr[n, :, :] = epf.extension

            post_state_arr[n, :] = epf.state
            post_cov_arr[n, :, :] = epf.cov
            post_ext_arr[n, :, :] = epf.extension
            continue

        epf.predict()
        prior_state_arr[n, :] = epf.state
        prior_cov_arr[n, :, :] = epf.cov
        prior_ext_arr[n, :, :] = epf.extension

        epf.correct(zs)
        post_state_arr[n, :] = epf.state
        post_cov_arr[n, :, :] = epf.cov
        post_ext_arr[n, :, :] = epf.extension
        print(n)

    print(epf)

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
        ax.scatter(trajs_meas[i][::3, 0], trajs_meas[i][::3, 1], marker='^', facecolors=None, edgecolors='k', s=5)
    for i in range(entries):
        for j in range(N):
            if j % 3 == 0:
                x, y = utils.ellip_point(trajs_meas[i][j, 0], trajs_meas[i][j, 1], post_ext_arr[j], 200)
                ax.plot(x, y, color='gray')
                x, y = utils.ellip_point(trajs_meas[i][j, 0], trajs_meas[i][j, 1], real_ellip[j], 200)
                ax.plot(x, y, color='green')
                ax.scatter(trajs_meas_ellip[j][:, 0], trajs_meas_ellip[j][:, 1], s=1)
    ax.plot(post_state_arr[:, 0], post_state_arr[:, 2], linewidth=0.8, label='post esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()


if __name__ == '__main__':
    KochEOT_test()
    FeldmannEOT_test()