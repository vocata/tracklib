#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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


def DMMF_test():
    x_dim, z_dim = 6, 2

    # generate trajectory
    start = np.array([100.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    T = 0.1
    traj = model.Trajectory2D(T, start)
    stages = []
    stages.append({'model': 'cv', 'len': 200, 'velocity': [10, 0]})
    stages.append({'model': 'ct', 'len': 200, 'omega': tlb.deg2rad(300) / (200 * T)})
    stages.append({'model': 'cv', 'len': 200, 'velocity': [None, None]})
    stages.append({'model': 'ct', 'len': 200, 'omega': tlb.deg2rad(60) / (200 * T)})
    stages.append({'model': 'cv', 'len': 200, 'velocity': [None, None]})
    stages.append({'model': 'ca', 'len': 200, 'acceleration': [10, 0]})
    traj.add_stage(stages)
    R = np.eye(2)
    traj_real, traj_meas = traj(R)
    N = len(traj)

    # add model
    trans_mat = np.array([[0.99, 0.005, 0.005], [0.005, 0.99, 0.005], [0.005, 0.005, 0.99]])

    # CV
    qx, qy = np.sqrt(1), np.sqrt(1)
    rx, ry = np.sqrt(1), np.sqrt(1)
    F = np.zeros((x_dim, x_dim))
    F[:4, :4] = model.F_poly_trans(1, 1, T)
    H = model.H_only_pos_meas(2, 1)
    L = np.eye(x_dim)
    M = np.eye(z_dim)
    Q = np.zeros((x_dim, x_dim))
    Q[:4, :4] = model.Q_dd_poly_proc_noise(1, 1, T, [qx, qy], 1)
    R = model.R_only_pos_meas_noise(1, [rx, ry])
    cv_kf = ft.KFilter(F, L, H, M, Q, R)

    # CA
    qx, qy = np.sqrt(1), np.sqrt(1)
    rx, ry = np.sqrt(1), np.sqrt(1)
    F = model.F_poly_trans(2, 1, T)
    H = model.H_only_pos_meas(2, 1)
    L = np.eye(x_dim)
    M = np.eye(z_dim)
    Q = model.Q_dd_poly_proc_noise(2, 1, T, [qx, qy])
    R = model.R_only_pos_meas_noise(1, [rx, ry])
    ca_kf = ft.KFilter(F, L, H, M, Q, R)

    # CT
    qx, qy = np.sqrt(1), np.sqrt(1)
    rx, ry = np.sqrt(1), np.sqrt(1)
    turn_rate = tlb.deg2rad(300) / (200 * T)
    F = np.zeros((x_dim, x_dim))
    F[:4, :4] = model.F_ct2D_trans(turn_rate, T)
    H = model.H_only_pos_meas(2, 1)
    Q = np.zeros((x_dim, x_dim))
    Q[:4, :4] = model.Q_ct2D_proc_noise(T, [qx, qy])
    ct_kf1 = ft.KFilter(F, L, H, M, Q, R)

    turn_rate = tlb.deg2rad(60) / (200 * T)
    F = np.zeros((x_dim, x_dim))
    F[:4, :4] = model.F_ct2D_trans(turn_rate, T)
    ct_kf2 = ft.KFilter(F, L, H, M, Q, R)

    mmf = ft.MMFilter()
    mmf.add_models([ct_kf1, ct_kf2], [1/2, 1/2])

    # dmmf = ft.GPB1Filter()
    dmmf = ft.GPB2Filter()
    # dmmf = ft.IMMFilter()
    dmmf.add_models([cv_kf, ca_kf, mmf], [1/3, 1/3, 1/3], trans_mat)

    post_state_arr = np.empty((x_dim, N - 1))
    prob_arr = np.empty((3, N - 1))     # 2 models
    subprob_arr = np.empty((2, N - 1))  # 2 submodels

    for n in range(-1, N - 1):
        if n == -1:
            # x_init, P_init = init.single_point_init(traj_meas[:, n + 1], R, 20)
            x_init, P_init = start, np.eye(x_dim)
            dmmf.init(x_init, P_init)
            continue
        dmmf.step(traj_meas[:, n + 1])

        post_state_arr[:, n] = dmmf.post_state
        prob_arr[:, n] = dmmf.probs()
        if isinstance(dmmf, ft.GPB2Filter):
            subprob_arr[:, n] = dmmf.models()[2][0].probs()
        else:
            subprob_arr[:, n] = dmmf.models()[2].probs()
    print(len(dmmf))
    print(dmmf)
    print(post_state_arr[:, -1])

    # trajectory
    _, ax = plt.subplots()
    ax.axis('equal')
    ax.scatter(traj_real[0, 0], traj_real[1, 0], s=120, c='r', marker='x', label='start')
    ax.plot(traj_real[0, :], traj_real[1, :], linewidth=0.8, label='real')
    ax.scatter(traj_meas[0, :], traj_meas[1, :], s=5, c='orange', label='meas')
    ax.plot(post_state_arr[0, :], post_state_arr[1, :], linewidth=0.8, label='esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()

    r = 3
    _, ax = plt.subplots()
    n = np.arange(N - 1)
    for i in range(r):
        ax.plot(n, prob_arr[i, :])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend([str(n) for n in range(r)])
    plt.show()

    sr = 2
    _, ax = plt.subplots()
    n = np.arange(N - 1)
    for i in range(sr):
        ax.plot(n, subprob_arr[i, :])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend([str(n) for n in range(sr)])
    plt.show()


if __name__ == '__main__':
    # gen_traj()
    DMMF_test()