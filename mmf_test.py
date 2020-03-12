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


def MMFilter_test():
    N, T = 200, 1
    model_n = 5

    Q, R = [], []
    for i in range(5):
        qx, qy = math.sqrt((i + 1)/100), math.sqrt((i + 2)/100)
        rx, ry = math.sqrt(i + 1), math.sqrt(i + 1)

        Q.append(np.diag([qx**2, qy**2]))
        R.append(np.diag([rx**2, ry**2]))
    F, L, H, M = map(lambda x: [x], ft.newton_sys(T, 2, 2))
    F, L, H, M = map(lambda x: x * model_n, (F, L, H, M))

    # initial state and error convariance
    x = utils.col([1, 2, 0.2, 0.3])
    P = 100 * np.eye(x.shape[0])

    filter = ft.MMFilter(F, L, H, M, Q, R)
    filter.init(x, P)

    x_arr = np.empty((4, N))
    z_arr = np.empty((2, N))
    x_esti_arr = np.empty((4, N))
    prob_arr = np.empty((model_n, N))

    for n in range(N):
        wx = np.random.normal(0, qx)
        wy = np.random.normal(0, qy)
        w = utils.col([wx, wy])
        vx = np.random.normal(0, rx)
        vy = np.random.normal(0, ry)
        v = utils.col([vx, vy])

        x = F[0] @ x + L[0] @ w
        z = H[0] @ x + M[0] @ v
        x_arr[:, n] = x[:, 0]
        z_arr[:, n] = z[:, 0]

        x_esti, prob = filter.step(z)
        x_esti_arr[:, n] = x_esti[:, 0]
        prob_arr[:, n] = np.array(prob)
    print(len(filter))

    # plot
    n = np.arange(N)
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, x_arr[0, :], linewidth=0.8)
    ax[0].plot(n, z_arr[0, :], '.')
    ax[0].plot(n, x_esti_arr[0, :], linewidth=0.8)
    ax[0].legend(['real', 'measurement', 'estimation'])
    ax[0].set_title('x state')
    ax[1].plot(n, x_arr[1, :], linewidth=0.8)
    ax[1].plot(n, z_arr[1, :], '.')
    ax[1].plot(n, x_esti_arr[1, :], linewidth=0.8)
    ax[1].legend(['realGo', 'measurement', 'estimation'])
    ax[1].set_title('y state')
    plt.show()

    _, ax = plt.subplots()
    for i in range(model_n):
        ax.plot(n, prob_arr[i, :])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend([str(n) for n in range(model_n)])
    plt.show()

    # trajectory
    _, ax = plt.subplots()
    ax.scatter(x_arr[0, 0], x_arr[1, 0], s=120, c='r', marker='x')
    ax.plot(x_arr[0, :], x_arr[1, :], linewidth=0.8)
    ax.plot(z_arr[0, :], z_arr[1, :], linewidth=0.8)
    ax.plot(x_esti_arr[0, :], x_esti_arr[1, :], linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['real', 'measurement', 'estimation'])
    ax.set_title('trajectory')
    plt.show()

if __name__ == '__main__':
    MMFilter_test()