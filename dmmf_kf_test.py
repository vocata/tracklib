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
    xdim, zdim = 6, 2

    # generate trajectory
    start = np.array([100.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    T = 0.1
    traj = model.Trajectory2D(T, start)
    stages = []
    stages.append({'model': 'cv', 'len': 200, 'velocity': [10, 0]})
    stages.append({'model': 'ct', 'len': 200, 'omega': tlb.deg2rad(360) / (200 * T)})
    stages.append({'model': 'ca', 'len': 200, 'acceleration': [10, 0]})
    traj.add_stage(stages)
    R = np.eye(2)
    traj_real, traj_meas = traj(R)
    N = len(traj)

    # CV
    qx, qy = np.sqrt(1.0), np.sqrt(1.0)
    rx, ry = np.sqrt(1.0), np.sqrt(1.0)
    F = np.zeros((xdim, xdim))
    F[:4, :4] = model.F_poly_trans(1, 1, T)
    H = model.H_only_pos_meas(2, 1)
    L = np.eye(xdim)
    M = np.eye(zdim)
    Q = np.zeros((xdim, xdim))
    Q[:4, :4] = model.Q_dd_poly_proc_noise(1, 1, T, [qx, qy], 1)
    R = model.R_only_pos_meas_noise(1, [rx, ry])
    cv_kf = ft.KFilter(F, L, H, M, Q, R, xdim, zdim)

    # CA
    qx, qy = np.sqrt(1.0), np.sqrt(1.0)
    rx, ry = np.sqrt(1.0), np.sqrt(1.0)
    F = model.F_poly_trans(2, 1, T)
    H = model.H_only_pos_meas(2, 1)
    L = np.eye(xdim)
    M = np.eye(zdim)
    Q = model.Q_dd_poly_proc_noise(2, 1, T, [qx, qy])
    R = model.R_only_pos_meas_noise(1, [rx, ry])
    ca_kf = ft.KFilter(F, L, H, M, Q, R, xdim, zdim)

    # CT
    qx, qy = np.sqrt(1.0), np.sqrt(1.0)
    rx, ry = np.sqrt(1.0), np.sqrt(1.0)
    turn_rate = tlb.deg2rad(360) / (200 * T)
    F = np.zeros((xdim, xdim))
    F[:4, :4] = model.F_ct2D_trans(turn_rate, T)
    H = model.H_only_pos_meas(2, 1)
    Q = np.zeros((xdim, xdim))
    Q[:4, :4] = model.Q_ct2D_proc_noise(T, [qx, qy])
    ct_kf = ft.KFilter(F, L, H, M, Q, R, xdim, zdim)

    r = 3
    # dmmf = ft.GPB1Filter()
    # dmmf = ft.GPB2Filter()
    dmmf = ft.IMMFilter()
    dmmf.add_models([cv_kf, ca_kf, ct_kf])

    post_state_arr = np.empty((xdim, N - 1))
    prob_arr = np.empty((r, N - 1))

    for n in range(-1, N - 1):
        if n == -1:
            # x_init, P_init = init.single_point_init(traj_meas[:, n + 1], R, 20)
            x_init, P_init = start, 10*np.eye(xdim)
            dmmf.init(x_init, P_init)
            continue
        dmmf.step(traj_meas[:, n + 1])

        post_state_arr[:, n] = dmmf.post_state
        prob_arr[:, n] = dmmf.probs()
    print(len(dmmf))
    print(dmmf)
    print(dmmf.prior_state)
    print(dmmf.post_state)

    # trajectory
    _, ax = plt.subplots()
    ax.axis('equal')
    ax.scatter(traj_real[0, 0], traj_real[1, 0], s=120, c='r', marker='x', label='start')
    ax.plot(traj_real[0, :], traj_real[1, :], linewidth=0.8, label='real')
    ax.scatter(traj_meas[0, :], traj_meas[1, :], s=5, c='orange', label='meas')
    ax.plot(post_state_arr[0, :], post_state_arr[1, :], linewidth=0.8, label='esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(['real', 'meas', 'esti'])
    ax.set_title('trajectory')
    plt.show()

    _, ax = plt.subplots()
    n = np.arange(N - 1)
    for i in range(r):
        ax.plot(n, prob_arr[i, :])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend([str(n) for n in range(r)])
    plt.show()


if __name__ == '__main__':
    # gen_traj()
    DMMF_test()