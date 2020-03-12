#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import tracklib.filter as ft
import tracklib.utils as utils
import matplotlib.pyplot as plt
'''
notes:
vector is preferably a column vector, otherwise
the program may yield uncertain result.
'''


def KFilter_test():
    N, T = 200, 1

    x_dim, z_dim = 4, 2
    qx, qy = math.sqrt(0.01), math.sqrt(0.02)
    rx, ry = math.sqrt(1), math.sqrt(1)

    Q = np.diag([qx**2, qy**2])
    R = np.diag([rx**2, ry**2])
    F, L, H, M = ft.newton_sys(T, 2, 2)

    # initial state and error convariance
    x = utils.col([1, 2, 0.2, 0.3])
    P = 100 * np.eye(x_dim)

    kf = ft.KFilter(F, L, H, M, Q, R)
    kf.init(x, P)

    x_arr = np.empty((x_dim, N))
    z_arr = np.empty((z_dim, N))
    x_pred_arr = np.empty((x_dim, N))
    x_up_arr = np.empty((x_dim, N))
    P_pred_arr = np.empty((x_dim, x_dim, N))
    P_up_arr = np.empty((x_dim, x_dim, N))
    innov_arr = np.empty((z_dim, N))
    inP_arr = np.empty((z_dim, z_dim, N))

    for n in range(N):
        wx = np.random.normal(0, qx)
        wy = np.random.normal(0, qy)
        w = utils.col([wx, wy])
        vx = np.random.normal(0, rx)
        vy = np.random.normal(0, ry)
        v = utils.col([vx, vy])

        x = F @ x + L @ w
        z = H @ x + M @ v
        x_arr[:, n] = x[:, 0]
        z_arr[:, n] = z[:, 0]
        x_pred, P_pred, x_up, P_up, K, innov, inP = kf.step(z)

        x_pred_arr[:, n] = x_pred[:, 0]
        x_up_arr[:, n] = x_up[:, 0]
        P_pred_arr[:, :, n] = P_pred
        P_up_arr[:, :, n] = P_up
        innov_arr[:, n] = innov[:, 0]
        inP_arr[:, :, n] = inP
    print(len(kf))
    print(kf)

    # plot
    n = np.arange(N)
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, x_arr[0, :], linewidth=0.8)
    ax[0].plot(n, z_arr[0, :], '.')
    ax[0].plot(n, x_pred_arr[0, :], linewidth=0.8)
    ax[0].plot(n, x_up_arr[0, :], linewidth=0.8)
    ax[0].legend(['real', 'measurement', 'prediction', 'estimation'])
    ax[0].set_title('x state')
    ax[1].plot(n, x_arr[1, :], linewidth=0.8)
    ax[1].plot(n, z_arr[1, :], '.')
    ax[1].plot(n, x_pred_arr[1, :], linewidth=0.8)
    ax[1].plot(n, x_up_arr[1, :], linewidth=0.8)
    ax[1].legend(['real', 'measurement', 'prediction', 'estimation'])
    ax[1].set_title('y state')
    plt.show()

    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, P_pred_arr[0, 0, :], linewidth=0.8)
    ax[0].plot(n, P_up_arr[0, 0, :], linewidth=0.8)
    ax[0].legend(['prediction', 'estimation'])
    ax[0].set_title('x error variance/mean square error')
    ax[1].plot(n, P_pred_arr[1, 1, :], linewidth=0.8)
    ax[1].plot(n, P_up_arr[1, 1, :], linewidth=0.8)
    ax[1].legend(['prediction', 'estimation'])
    ax[1].set_title('y error variance/mean square error')
    plt.show()

    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, innov_arr[0, :], linewidth=0.8)
    ax[0].set_title('x innovation')
    ax[1].plot(n, innov_arr[1, :], linewidth=0.8)
    ax[1].set_title('y innovation')
    plt.show()
    print('mean of x innovation: %f' % innov_arr[0, :].mean())
    print('mean of y innovation: %f' % innov_arr[1, :].mean())

    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, inP_arr[0, 0, :], linewidth=0.8)
    ax[0].set_title('x innovation variance')
    ax[1].plot(n, inP_arr[1, 1, :], linewidth=0.8)
    ax[1].set_title('y innovation variance')
    plt.show()

    print('The shape of kalman gain: %s' % str(K.shape))

    # trajectory
    _, ax = plt.subplots()
    ax.scatter(x_arr[0, 0], x_arr[1, 0], s=120, c='r', marker='x')
    ax.plot(x_arr[0, :], x_arr[1, :], linewidth=0.8)
    ax.plot(z_arr[0, :], z_arr[1, :], linewidth=0.8)
    ax.plot(x_up_arr[0, :], x_up_arr[1, :], linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['real', 'measurement', 'estimation'])
    ax.set_title('trajectory')
    plt.show()


if __name__ == '__main__':
    KFilter_test()