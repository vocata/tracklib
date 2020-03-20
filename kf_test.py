#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
import tracklib.filter as ft
import tracklib.utils as utils
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
'''
notes:
vector is preferably a column vector, otherwise
the program may yield uncertain result.
'''


def KFilter_test():
    N, T = 200, 1

    x_dim, z_dim = 4, 2
    qx, qy = np.sqrt(0.01), np.sqrt(0.02)
    rx, ry = np.sqrt(1), np.sqrt(1)

    Q = np.diag([qx**2, qy**2])
    R = np.diag([rx**2, ry**2])
    F, L, H, M = ft.newton_sys(T, 2, 2)

    # initial state and error convariance
    x = utils.col([1, 2, 0.2, 0.3])
    P = 100 * np.eye(x_dim)

    kf = ft.KFilter(F, L, H, M, Q, R)
    kf.init(x, P)

    state_arr = np.empty((x_dim, N))
    measure_arr = np.empty((z_dim, N))
    prior_state_arr = np.empty((x_dim, N))
    post_state_arr = np.empty((x_dim, N))
    prior_cov_arr = np.empty((x_dim, x_dim, N))
    post_cov_arr = np.empty((x_dim, x_dim, N))
    innov_arr = np.empty((z_dim, N))
    innov_cov_arr = np.empty((z_dim, z_dim, N))

    for n in range(N):
        wx = np.random.normal(0, qx)
        wy = np.random.normal(0, qy)
        w = utils.col([wx, wy])
        vx = np.random.normal(0, rx)
        vy = np.random.normal(0, ry)
        v = utils.col([vx, vy])

        x = F @ x + L @ w
        z = H @ x + M @ v
        state_arr[:, n] = x[:, 0]
        measure_arr[:, n] = z[:, 0]
        kf.step(z)
        prior_state, prior_cov = kf.prior_state, kf.prior_cov
        post_state, post_cov = kf.post_state, kf.post_cov
        innov, innov_cov = kf.innov, kf.innov_cov
        gain = kf.gain

        prior_state_arr[:, n] = prior_state[:, 0]
        post_state_arr[:, n] = post_state[:, 0]
        prior_cov_arr[:, :, n] = prior_cov
        post_cov_arr[:, :, n] = post_cov
        innov_arr[:, n] = innov[:, 0]
        innov_cov_arr[:, :, n] = innov_cov
    print(len(kf))
    print(kf)

    # plot
    n = np.arange(N)
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, state_arr[0, :], linewidth=0.8)
    ax[0].plot(n, measure_arr[0, :], '.')
    ax[0].plot(n, prior_state_arr[0, :], linewidth=0.8)
    ax[0].plot(n, post_state_arr[0, :], linewidth=0.8)
    ax[0].legend(['real', 'measurement', 'prediction', 'estimation'])
    ax[0].set_title('x state')
    ax[1].plot(n, state_arr[1, :], linewidth=0.8)
    ax[1].plot(n, measure_arr[1, :], '.')
    ax[1].plot(n, prior_state_arr[1, :], linewidth=0.8)
    ax[1].plot(n, post_state_arr[1, :], linewidth=0.8)
    ax[1].legend(['real', 'measurement', 'prediction', 'estimation'])
    ax[1].set_title('y state')
    plt.show()

    print('x prior error variance {}'.format(prior_cov_arr[0, 0, -1]))
    print('x posterior error variance {}'.format(post_cov_arr[0, 0, -1]))
    print('y prior error variance {}'.format(prior_cov_arr[1, 1, -1]))
    print('y posterior error variance {}'.format(post_cov_arr[1, 1, -1]))
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, prior_cov_arr[0, 0, :], linewidth=0.8)
    ax[0].plot(n, post_cov_arr[0, 0, :], linewidth=0.8)
    ax[0].legend(['prediction', 'estimation'])
    ax[0].set_title('x error variance/mean square error')
    ax[1].plot(n, prior_cov_arr[1, 1, :], linewidth=0.8)
    ax[1].plot(n, post_cov_arr[1, 1, :], linewidth=0.8)
    ax[1].legend(['prediction', 'estimation'])
    ax[1].set_title('y error variance/mean square error')
    plt.show()

    print('mean of x innovation: {}'.format(innov_arr[0, :].mean()))
    print('mean of y innovation: {}'.format(innov_arr[1, :].mean()))
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, innov_arr[0, :], linewidth=0.8)
    ax[0].set_title('x innovation')
    ax[1].plot(n, innov_arr[1, :], linewidth=0.8)
    ax[1].set_title('y innovation')
    plt.show()

    print('x innovation variance {}'.format(innov_cov_arr[0, 0, -1]))
    print('y innovation variance {}'.format(innov_cov_arr[1, 1, -1]))
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, innov_cov_arr[0, 0, :], linewidth=0.8)
    ax[0].set_title('x innovation variance')
    ax[1].plot(n, innov_cov_arr[1, 1, :], linewidth=0.8)
    ax[1].set_title('y innovation variance')
    plt.show()

    print('Kalman gain:\n{}'.format(gain))

    # trajectory
    _, ax = plt.subplots()
    ax.scatter(state_arr[0, 0], state_arr[1, 0], s=120, c='r', marker='x')
    ax.plot(state_arr[0, :], state_arr[1, :], linewidth=0.8)
    ax.plot(measure_arr[0, :], measure_arr[1, :], linewidth=0.8)
    ax.plot(post_state_arr[0, :], post_state_arr[1, :], linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['real', 'measurement', 'estimation'])
    ax.set_title('trajectory')
    plt.show()

    # trajectory amination
    # _, ax = plt.subplots()
    # ax.scatter(state_arr[0, 0], state_arr[1, 0], s=120, c='r', marker='x')
    # ax.plot(state_arr[0, :], state_arr[1, :], linewidth=0.8)
    # ax.plot(measure_arr[0, :], measure_arr[1, :], linewidth=0.8)
    # ax.plot(prior_state_arr[0, :], prior_state_arr[1, :], linewidth=0.8)
    # ax.plot(post_state_arr[0, :], post_state_arr[1, :], linewidth=0.8)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.legend(['real', 'measurement', 'prior esti', 'post esti'])
    # ax.set_title('trajectory')
    # plt.ion()
    # for i in range(N):
    #     prior_state = prior_state_arr[:2, i]
    #     prior_cov = prior_cov_arr[:2, :2, i]
    #     post_state = post_state_arr[:2, i]
    #     post_cov = post_cov_arr[:2, :2, i]
    #     origin = prior_state
    #     d, v = lg.eigh(prior_cov)
    #     width = 2 * np.sqrt(d[0])
    #     height = 2 * np.sqrt(d[1])
    #     angle = np.rad2deg(np.log(complex(v[0, 0], v[1, 1])).imag)
    #     e = Ellipse(origin, width, height, angle)
    #     e.set_facecolor('white')
    #     e.set_edgecolor('black')
    #     ax.add_patch(e)
    #     plt.pause(0.2)
    #     e.remove()
    #     origin = post_state
    #     d, v = lg.eigh(post_cov)
    #     width = 2 * np.sqrt(d[0])
    #     height = 2 * np.sqrt(d[1])
    #     angle = np.rad2deg(np.log(complex(v[0, 0], v[1, 1])).imag)
    #     e = Ellipse(origin, width, height, angle)
    #     e.set_facecolor('white')
    #     e.set_edgecolor('red')
    #     ax.add_patch(e)
    #     plt.pause(0.2)
    #     e.remove()
    # plt.ioff()
    # plt.show()


if __name__ == '__main__':
    KFilter_test()