#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tracklib.filter as ft
import matplotlib.pyplot as plt
'''
notes:
vector is preferably a column vector, otherwise
the program may yield uncertain result.
'''


def SSFilter_test():
    N, T = 200, 1

    x_dim, z_dim = 4, 2
    qx, qy = np.sqrt(0.01), np.sqrt(0.02)
    rx, ry = np.sqrt(1), np.sqrt(1)

    Q = np.diag([qx**2, qy**2])
    R = np.diag([rx**2, ry**2])
    F, L, H, M = ft.newton_sys(T, 2, 2)

    # initial state and error convariance
    x = np.array([1, 2, 0.2, 0.3])
    P = 100 * np.eye(x_dim)

    ssf = ft.SSFilter(F, L, H, M, Q, R)
    ssf.init(x, P)

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
        w = np.array([wx, wy])
        vx = np.random.normal(0, rx)
        vy = np.random.normal(0, ry)
        v = np.array([vx, vy])

        x = F @ x + L @ w
        z = H @ x + M @ v
        state_arr[:, n] = x
        measure_arr[:, n] = z
        ssf.step(z)

        prior_state, prior_cov = ssf.prior_state, ssf.prior_cov
        post_state, post_cov = ssf.post_state, ssf.post_cov
        innov, innov_cov = ssf.innov, ssf.innov_cov
        gain = ssf.gain

        prior_state_arr[:, n] = prior_state
        post_state_arr[:, n] = post_state
        prior_cov_arr[:, :, n] = prior_cov
        post_cov_arr[:, :, n] = post_cov
        innov_arr[:, n] = innov
        innov_cov_arr[:, :, n] = innov_cov
    print(len(ssf))
    print(ssf)

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


if __name__ == '__main__':
    SSFilter_test()