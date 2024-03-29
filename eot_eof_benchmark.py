#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as io
import scipy.linalg as lg
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import tracklib.utils as utils
import matplotlib.pyplot as plt


def plot_ellipse(ax, x0, y0, C, N, *args, **kwargs):
    x, y = utils.ellip_point(x0, y0, C, N)
    ax.plot(x, y, *args, **kwargs)


def GTT_Koch_test(epoch):
    data = io.loadmat('eot_data.mat')
    trajs_meas = data['trajs_meas'][epoch]
    trajs_state = data['trajs_state']
    real_ellip = data['real_ellip']

    N = trajs_state.shape[0]
    T = 10
    tau = 4 * T
    df = 60

    axis = 2
    zdim, xdim = 2, 4
    sigma_w = 0.05
    sigma_v = [50., 50.]

    F = model.F_cv(1, T)
    H = model.H_cv(1)
    Q = model.Q_cv_dc(1, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    eopf = ft.KochEOFilter(F, H, Q, T, tau)

    prior_state_arr = np.empty((N, xdim))
    prior_cov_arr = np.empty((N, xdim, xdim))
    post_state_arr = np.empty((N, xdim))
    post_cov_arr = np.empty((N, xdim, xdim))
    prior_ext_arr = np.empty((N, zdim, zdim))
    post_ext_arr = np.empty((N, zdim, zdim))

    for n in range(N):
        if n == 0:
            z_mean = np.mean(trajs_meas[n], axis=0)
            ellip = 100**2 * np.eye(2)
            x_init, _ = init.cv_init(z_mean, R, (10, 10))
            P_init = np.diag([1., 1.])
            x_init[1], x_init[3] = 14, -14
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

        if len(trajs_meas[n]) != 0:
            eopf.correct(trajs_meas[n])
            post_state_arr[n, :] = eopf.state
            post_cov_arr[n, :, :] = eopf.cov
            post_ext_arr[n, :, :] = eopf.extension
        # print(n)

    return post_state_arr, post_ext_arr, trajs_state, real_ellip
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
    for i in range(N):
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


def GTT_Feldmann_test(epoch):
    data = io.loadmat('eot_data.mat')
    trajs_meas = data['trajs_meas'][epoch]
    trajs_state = data['trajs_state']
    real_ellip = data['real_ellip']

    N = trajs_state.shape[0]
    T = 10
    df = 60
    tau = 4 * T

    axis = 2
    zdim, xdim = 2, 4
    sigma_w = 5
    sigma_v = [50., 50.]

    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    Q = model.Q_cv_dc(axis, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    eopf = ft.FeldmannEOFilter(F, H, Q, R, T, tau)

    prior_state_arr = np.empty((N, xdim))
    prior_cov_arr = np.empty((N, xdim, xdim))
    post_state_arr = np.empty((N, xdim))
    post_cov_arr = np.empty((N, xdim, xdim))
    prior_ext_arr = np.empty((N, zdim, zdim))
    post_ext_arr = np.empty((N, zdim, zdim))

    for n in range(N):
        if n == 0:
            z_mean = np.mean(trajs_meas[n], axis=0)
            ellip = 100**2 * np.eye(2)
            x_init, _ = init.cv_init(z_mean, R, (10, 10))
            P_init = np.kron(ellip, np.diag([1., 1.]))
            x_init[1], x_init[3] = 14, -14
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

        if len(trajs_meas[n]) != 0:
            eopf.correct(trajs_meas[n])
            post_state_arr[n, :] = eopf.state
            post_cov_arr[n, :, :] = eopf.cov
            post_ext_arr[n, :, :] = eopf.extension
        # print(n)

    return post_state_arr, post_ext_arr, trajs_state, real_ellip
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
    for i in range(N):
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

def GTT_Lan_test(epoch):
    data = io.loadmat('eot_data.mat')
    trajs_meas = data['trajs_meas'][epoch]
    trajs_state = data['trajs_state']
    real_ellip = data['real_ellip']

    N = trajs_state.shape[0]
    T = 10
    delta = 50
    df = 60

    axis = 2
    zdim, xdim = 2, 4
    sigma_w = 0.05
    sigma_v = [50., 50.]

    F = model.F_cv(1, T)
    H = model.H_cv(1)
    Q = model.Q_cv_dc(1, T, sigma_w)
    R = model.R_cv(axis, sigma_v)

    eopf = ft.LanEOFilter(F, H, Q, R, delta)

    prior_state_arr = np.empty((N, xdim))
    prior_cov_arr = np.empty((N, xdim, xdim))
    post_state_arr = np.empty((N, xdim))
    post_cov_arr = np.empty((N, xdim, xdim))
    prior_ext_arr = np.empty((N, zdim, zdim))
    post_ext_arr = np.empty((N, zdim, zdim))

    for n in range(N):
        if n == 0:
            z_mean = np.mean(trajs_meas[n], axis=0)
            ellip = 100**2 * np.eye(2)
            x_init, _ = init.cv_init(z_mean, R, (10, 10))
            P_init = np.diag([1., 1.])
            x_init[1], x_init[3] = 14, -14
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

        if len(trajs_meas[n]) != 0:
            eopf.correct(trajs_meas[n])
            post_state_arr[n, :] = eopf.state
            post_cov_arr[n, :, :] = eopf.cov
            post_ext_arr[n, :, :] = eopf.extension
        # print(n)

    return post_state_arr, post_ext_arr, trajs_state, real_ellip
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
    for i in range(N):
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
    # koch approach
    koch_state, koch_ext = 0., 0.
    _, _, trajs_state, real_ellip = GTT_Koch_test(0)
    real_pos = trajs_state[:, [0, 3]]
    real_vel = trajs_state[:, [1, 4]]
    koch_ext_err, koch_pos_err, koch_vel_err = 0., 0., 0.
    for i in range(1000):
        state, ext, _, _ = GTT_Koch_test(i)
        print(i)
        koch_state = i * koch_state / (i + 1) + state / (i + 1)
        koch_ext = i * koch_ext / (i + 1) + ext / (i + 1)

        koch_ext_err = i * koch_ext_err / (i + 1) + np.trace((ext - real_ellip) @ (ext - real_ellip), axis1=1, axis2=2) / (i + 1)
        koch_pos_err = i * koch_pos_err / (i + 1) + lg.norm(real_pos - state[:, [0, 2]], axis=1)**2 / (i + 1)
        koch_vel_err = i * koch_vel_err / (i + 1) + lg.norm(real_vel - state[:, [1, 3]], axis=1)**2 / (i + 1)
    koch_ext_rmse = np.sqrt(koch_ext_err)
    koch_pos_rmse = np.sqrt(koch_pos_err)
    koch_vel_rmse = np.sqrt(koch_vel_err)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(trajs_state[::2, 0], trajs_state[::2, 3], marker='^', facecolors=None, edgecolors='k', s=5)
    ax.plot(koch_state[::2, 0], koch_state[::2, 2])
    for i in range(len(koch_state)):
        if i % 2 == 0:
            x, y = utils.ellip_point(koch_state[i, 0], koch_state[i, 2], koch_ext[i], 200)
            ax.plot(x, y, color='gray')
    plt.show(block=False)

    # feldmann approach
    feldmann_state, feldmann_ext = 0., 0.
    _, _, trajs_state, real_ellip = GTT_Feldmann_test(0)
    real_pos = trajs_state[:, [0, 3]]
    real_vel = trajs_state[:, [1, 4]]
    feldmann_ext_err, feldmann_pos_err, feldmann_vel_err = 0., 0., 0.
    for i in range(1000):
        state, ext, _, _ = GTT_Feldmann_test(i)
        print(i)
        feldmann_state = i * feldmann_state / (i + 1) + state / (i + 1)
        feldmann_ext = i * feldmann_ext / (i + 1) + ext / (i + 1)

        feldmann_ext_err = i * feldmann_ext_err / (i + 1) + np.trace((ext - real_ellip) @ (ext - real_ellip), axis1=1, axis2=2) / (i + 1)
        feldmann_pos_err = i * feldmann_pos_err / (i + 1) + lg.norm(real_pos - state[:, [0, 2]], axis=1)**2 / (i + 1)
        feldmann_vel_err = i * feldmann_vel_err / (i + 1) + lg.norm(real_vel - state[:, [1, 3]], axis=1)**2 / (i + 1)
    feldmann_ext_rmse = np.sqrt(feldmann_ext_err)
    feldmann_pos_rmse = np.sqrt(feldmann_pos_err)
    feldmann_vel_rmse = np.sqrt(feldmann_vel_err)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(trajs_state[::2, 0], trajs_state[::2, 3], marker='^', facecolors=None, edgecolors='k', s=5)
    ax.plot(feldmann_state[::2, 0], feldmann_state[::2, 2])
    for i in range(len(feldmann_state)):
        if i % 2 == 0:
            x, y = utils.ellip_point(feldmann_state[i, 0], feldmann_state[i, 2], feldmann_ext[i], 200)
            ax.plot(x, y, color='gray')
    plt.show(block=False)

    # lan approach
    lan_state, lan_ext = 0., 0.
    _, _, trajs_state, real_ellip = GTT_Lan_test(0)
    real_pos = trajs_state[:, [0, 3]]
    real_vel = trajs_state[:, [1, 4]]
    lan_ext_err, lan_pos_err, lan_vel_err = 0., 0., 0.
    for i in range(1000):
        state, ext, _, _ = GTT_Lan_test(i)
        print(i)
        lan_state = i * lan_state / (i + 1) + state / (i + 1)
        lan_ext = i * lan_ext / (i + 1) + ext / (i + 1)

        lan_ext_err = i * lan_ext_err / (i + 1) + np.trace((ext - real_ellip) @ (ext - real_ellip), axis1=1, axis2=2) / (i + 1)
        lan_pos_err = i * lan_pos_err / (i + 1) + lg.norm(real_pos - state[:, [0, 2]], axis=1)**2 / (i + 1)
        lan_vel_err = i * lan_vel_err / (i + 1) + lg.norm(real_vel - state[:, [1, 3]], axis=1)**2 / (i + 1)
    lan_ext_rmse = np.sqrt(lan_ext_err)
    lan_pos_rmse = np.sqrt(lan_pos_err)
    lan_vel_rmse = np.sqrt(lan_vel_err)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(trajs_state[::2, 0], trajs_state[::2, 3], marker='^', facecolors=None, edgecolors='k', s=5)
    ax.plot(lan_state[::2, 0], lan_state[::2, 2])
    for i in range(len(lan_state)):
        if i % 2 == 0:
            x, y = utils.ellip_point(lan_state[i, 0], lan_state[i, 2], lan_ext[i], 200)
            ax.plot(x, y, color='gray')
    plt.show(block=False)

    # rmse
    t = np.arange(0, len(koch_ext_rmse) * 10, 10)
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(t, koch_ext_rmse, t, feldmann_ext_rmse, t, lan_ext_rmse)
    ax.legend(['koch', 'feldmann', 'lan'])
    ax = fig.add_subplot(312)
    ax.plot(t, koch_pos_rmse, t, feldmann_pos_rmse, t, lan_pos_rmse)
    ax.legend(['koch', 'feldmann', 'lan'])
    ax = fig.add_subplot(313)
    ax.plot(t, koch_vel_rmse, t, feldmann_vel_rmse, t, lan_vel_rmse)
    ax.legend(['koch', 'feldmann', 'lan'])
    plt.show()

    io.savemat(
        'eot_monte_carlo.mat', {
            'koch_ext_rmse': koch_ext_rmse,
            'koch_pos_rmse': koch_pos_rmse,
            'koch_vel_rmse': koch_vel_rmse,
            'feldmann_ext_rmse': feldmann_ext_rmse,
            'feldmann_pos_rmse': feldmann_pos_rmse,
            'feldmann_vel_rmse': feldmann_vel_rmse,
            'lan_ext_rmse': lan_ext_rmse,
            'lan_pos_rmse': lan_pos_rmse,
            'lan_vel_rmse': lan_vel_rmse
        })
