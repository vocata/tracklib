#! /usr/bin/env python3
# -*- coding: utf-8 -*-

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
    model_n = 3

    x_dim, z_dim = 4, 2
    mmf = ft.MMFilter()
    for i in range(model_n):
        qx, qy = np.sqrt((i + 1) / 10), np.sqrt((i + 1) / 10)
        rx, ry = np.sqrt(i + 1), np.sqrt(i + 1)

        Q = np.diag([qx**2, qy**2])
        R = np.diag([rx**2, ry**2])
        F, L, H, M = ft.newton_sys(T, 2, 2)
        model = ft.KFilter(F, L, H, M, Q, R)
        mmf.add_model(model, 1 / model_n)

    # initial state and error convariance
    x = np.array([1, 2, 0.2, 0.3])
    P = 100 * np.eye(x_dim)

    mmf.init(x, P)

    state_arr = np.empty((x_dim, N))
    measure_arr = np.empty((z_dim, N))
    weight_state_arr = np.empty((x_dim, N))
    maxprob_state_arr = np.empty((x_dim, N))
    prob_arr = np.empty((model_n, N))

    for n in range(N):
        wx = np.random.normal(0, qx)
        wy = np.random.normal(0, qy)
        w = np.array([wx, wy])
        vx = np.random.normal(0, rx)
        vy = np.random.normal(0, ry)
        v = np.array([vx, vy])

        x = F @ x + L @ w
        z = H @ x + M @ v
        state_arr[:, n] = x
        measure_arr[:, n] = z
        mmf.step(z)
        
        weight_state_arr[:, n] = mmf.weight_state
        maxprob_state_arr[:, n] = mmf.maxprob_state
        prob_arr[:, n] = mmf.prob
    print(len(mmf))
    print(mmf)

    # plot
    n = np.arange(N)
    _, ax = plt.subplots(2, 1)
    ax[0].plot(n, state_arr[0, :], linewidth=0.8)
    ax[0].plot(n, measure_arr[0, :], '.')
    ax[0].plot(n, weight_state_arr[0, :], linewidth=0.8)
    ax[0].plot(n, maxprob_state_arr[0, :], linewidth=0.8)
    ax[0].legend(['real', 'measurement', 'weighted esti', 'max prob esti'])
    ax[0].set_title('x state')
    ax[1].plot(n, state_arr[1, :], linewidth=0.8)
    ax[1].plot(n, measure_arr[1, :], '.')
    ax[1].plot(n, weight_state_arr[1, :], linewidth=0.8)
    ax[1].plot(n, maxprob_state_arr[1, :], linewidth=0.8)
    ax[1].legend(['real', 'measurement', 'weighted esti', 'max prob esti'])
    ax[1].set_title('y state weighted estimation')
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
    ax.scatter(state_arr[0, 0], state_arr[1, 0], s=120, c='r', marker='x')
    ax.plot(state_arr[0, :], state_arr[1, :], linewidth=0.8)
    ax.plot(measure_arr[0, :], measure_arr[1, :], linewidth=0.8)
    ax.plot(weight_state_arr[0, :], weight_state_arr[1, :], linewidth=0.8)
    ax.plot(maxprob_state_arr[0, :], maxprob_state_arr[1, :], linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['real', 'measurement', 'weighted esti', 'max prob esti'])
    ax.set_title('trajectory')
    plt.show()


if __name__ == '__main__':
    MMFilter_test()