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
    T = 0.1
    xdim, zdim = 6, 2

    # generate trajectory
    start = np.array([100.0, 0.0, 0.0, 100.0, 0.0, 0.0], dtype=float)
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
    sigma_w = [np.sqrt(1.0), np.sqrt(1.0)]
    sigma_v = [np.sqrt(1.0), np.sqrt(1.0)]
    F = np.zeros((xdim, xdim))
    F[:2, :2] = model.F_poly_trans(1, 0, T)
    F[3:5, 3:5] = model.F_poly_trans(1, 0, T)
    H = model.H_only_pos_meas(2, 1)
    L = np.eye(xdim)
    M = np.eye(zdim)
    Q = np.zeros((xdim, xdim))
    Q[:2, :2] = model.Q_dd_poly_proc_noise(1, 0, T, sigma_w[0], 1)
    Q[3:5, 3:5] = model.Q_dd_poly_proc_noise(1, 0, T, sigma_w[1], 1)
    R = model.R_only_pos_meas_noise(1, sigma_v)
    cv_kf = ft.KFilter(F, L, H, M, Q, R, xdim, zdim)

    # CA
    sigma_w = [np.sqrt(1.0), np.sqrt(1.0)]
    sigma_v = [np.sqrt(1.0), np.sqrt(1.0)]
    F = model.F_poly_trans(2, 1, T)
    H = model.H_only_pos_meas(2, 1)
    L = np.eye(xdim)
    M = np.eye(zdim)
    Q = model.Q_dd_poly_proc_noise(2, 1, T, sigma_w)
    R = model.R_only_pos_meas_noise(1, sigma_v)
    ca_kf = ft.KFilter(F, L, H, M, Q, R, xdim, zdim)

    # CT
    sigma_w = [np.sqrt(1.0), np.sqrt(1.0)]
    sigma_v = [np.sqrt(1.0), np.sqrt(1.0)]
    turn_rate = tlb.deg2rad(360) / (200 * T)
    F = np.zeros((xdim, xdim))
    Fct = model.F_ct2D_trans(turn_rate, T)
    F[:2, :2] = Fct[:2, :2]
    F[:2, 3:5] = Fct[:2, 2:]
    F[3:5, :2] = Fct[2:, :2]
    F[3:5, 3:5] = Fct[2:, 2:]
    H = model.H_only_pos_meas(2, 1)
    L = np.eye(xdim)
    M = np.eye(zdim)
    Q = np.zeros((xdim, xdim))
    Qct = model.Q_ct2D_proc_noise(T, sigma_w)
    Q[:2, :2] = Qct[:2, :2]
    Q[3:5, 3:5] = Qct[2:, 2:]
    R = model.R_only_pos_meas_noise(1, sigma_v)
    ct_kf = ft.KFilter(F, L, H, M, Q, R, xdim, zdim)

    r = 3
    
    # dmmf = ft.GPB1Filter()
    # dmmf = ft.GPB2Filter()
    dmmf = ft.IMMFilter()
    dmmf.add_models([cv_kf, ca_kf, ct_kf])

    x_init, P_init = start, np.eye(xdim)
    dmmf.init(x_init, P_init)

    post_state_arr = np.empty((xdim, N))
    prob_arr = np.empty((r, N))

    for n in range(N):
        dmmf.step(traj_meas[:, n])

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
    ax.plot(post_state_arr[0, :], post_state_arr[3, :], linewidth=0.8, label='esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()

    _, ax = plt.subplots()
    n = np.arange(N)
    for i in range(r):
        ax.plot(n, prob_arr[i, :])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.legend([str(n) for n in range(r)])
    plt.show()


if __name__ == '__main__':
    # gen_traj()
    DMMF_test()