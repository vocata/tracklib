#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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


def gen_traj(plot=True):
    # generate trajectory
    stage1 = 100    # straight right
    stage2 = stage1 + 50     # turn counterclockwise
    stage3 = stage2 + 200    # straight up
    stage4 = stage3 + 50     # turn clockwise
    stage5 = stage4 + 100
    N, T = stage5, 0.1

    x_dim, z_dim = 4, 2
    qx, qy = np.sqrt(0), np.sqrt(0)
    rx, ry = np.sqrt(10), np.sqrt(10)

    traj_real = np.zeros((x_dim, N))
    traj_meas = np.zeros((z_dim, N))
    H = model.H_only_pos_meas(1, 1)
    L = np.eye(x_dim)
    M = np.eye(z_dim)
    Q = model.Q_dd_poly_proc_noise(1, 1, T, [qx, qy])
    R = model.R_only_pos_meas_noise(1, [rx, ry])

    # start point
    x = np.array([10, 10, 10, 0])
    for n in range(N):
        if n < stage1:
            F = model.F_poly_trans(1, 1, T)
            w = tlb.multi_normal(0, Q)
            v = tlb.multi_normal(0, R)
            x = F @ x + L @ w
            z = H @ x + M @ v
            traj_real[:, n] = x
            traj_meas[:, n] = z
        elif n < stage2:
            turn_rate = np.pi / 2 / ((stage2 - stage1) * T)
            F = model.F_ct2D_trans(turn_rate, T)
            w = tlb.multi_normal(0, Q)
            v = tlb.multi_normal(0, R)
            x = F @ x + L @ w
            z = H @ x + M @ v
            traj_real[:, n] = x
            traj_meas[:, n] = z
        elif n < stage3:
            F = model.F_poly_trans(1, 1, T)
            w = tlb.multi_normal(0, Q)
            v = tlb.multi_normal(0, R)
            x = F @ x + L @ w
            z = H @ x + M @ v
            traj_real[:, n] = x
            traj_meas[:, n] = z
        elif n < stage4:
            turn_rate = -np.pi / 2 / ((stage4 - stage3) * T)
            F = model.F_ct2D_trans(turn_rate, T)
            w = tlb.multi_normal(0, Q)
            v = tlb.multi_normal(0, R)
            x = F @ x + L @ w
            z = H @ x + M @ v
            traj_real[:, n] = x
            traj_meas[:, n] = z
        else:
            F = model.F_poly_trans(1, 1, T)
            w = tlb.multi_normal(0, Q)
            v = tlb.multi_normal(0, R)
            x = F @ x + L @ w
            z = H @ x + M @ v
            traj_real[:, n] = x
            traj_meas[:, n] = z


    if plot:
        # trajectory
        _, ax = plt.subplots()
        ax.scatter(traj_real[0, 0], traj_real[1, 0], s=120, c='r', marker='x', label='start')
        ax.plot(traj_real[0, :], traj_real[1, :], linewidth=0.8, label='real')
        ax.scatter(traj_meas[0, :], traj_meas[1, :], s=5, c='orange', label='measurement')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.set_title('trajectory')
        plt.show()

    return traj_real, traj_meas, N, T, x_dim, z_dim


def DMMF_test():
    traj_real, traj_meas, N, T, x_dim, z_dim = gen_traj(False)

    trans_mat = np.array([[0.9, 0.1], [0.1, 0.9]])

    qx, qy = np.sqrt(0.005), np.sqrt(0.005)
    rx, ry = np.sqrt(5), np.sqrt(5)
    F = model.F_poly_trans(1, 1, T)
    H = model.H_only_pos_meas(1, 1)
    L = np.eye(x_dim)
    M = np.eye(z_dim)
    Q = model.Q_dd_poly_proc_noise(1, 1, T, [qx, qy])
    R = model.R_only_pos_meas_noise(1, [rx, ry])
    kf = ft.KFilter(F, L, H, M, Q, R)

    qx, qy = np.sqrt(0.001), np.sqrt(0.001)
    rx, ry = np.sqrt(5), np.sqrt(5)
    turn_rate = np.pi / 2 / (50 * T)
    F = model.F_ct2D_trans(turn_rate, T)
    Q = model.Q_ct2D_proc_noise(T, [qx, qy])
    subkf1 = ft.KFilter(F, L, H, M, Q, R)
    F = model.F_ct2D_trans(-turn_rate, T)
    subkf2 = ft.KFilter(F, L, H, M, Q, R)
    mmf = ft.MMFilter()
    mmf.add_models([subkf1, subkf2], [1/2, 1/2])

    # dmmf = ft.GPB1Filter()
    # dmmf = ft.GPB2Filter()
    dmmf = ft.IMMFilter()
    dmmf.add_models([kf, mmf], [1/2, 1/2], trans_mat)

    post_state_arr = np.empty((x_dim, N - 1))
    prob_arr = np.empty((2, N - 1))     # 2 models
    subprob_arr = np.empty((2, N - 1))  # 2 submodels

    for n in range(-1, N - 1):
        if n == -1:
            x_init, P_init = init.single_point_init(traj_meas[:, n + 1], R, 20)
            dmmf.init(x_init, P_init)
            continue
        print(n)
        dmmf.step(traj_meas[:, n + 1])

        post_state_arr[:, n] = dmmf.post_state
        prob_arr[:, n] = dmmf.probs
        subprob_arr[:, n] = dmmf.models[1].probs

    # trajectory
    _, ax = plt.subplots()
    ax.scatter(traj_real[0, 0], traj_real[1, 0], s=120, c='r', marker='x', label='start')
    ax.plot(traj_real[0, :], traj_real[1, :], linewidth=0.8, label='real')
    ax.scatter(traj_meas[0, :], traj_meas[1, :], s=5, c='orange', label='measurement')
    ax.plot(post_state_arr[0, :], post_state_arr[1, :], linewidth=0.8, label='estimate')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()

    _, ax = plt.subplots()
    n = np.arange(N - 1)
    for i in range(2):
        ax.plot(n, prob_arr[i, :])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend([str(n) for n in range(2)])
    plt.show()

    _, ax = plt.subplots()
    n = np.arange(N - 1)
    for i in range(2):
        ax.plot(n, subprob_arr[i, :])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend([str(n) for n in range(2)])
    plt.show()


if __name__ == '__main__':
    # gen_traj()
    DMMF_test()