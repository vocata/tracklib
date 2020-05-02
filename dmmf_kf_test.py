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
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    traj = model.Trajectory2D(T, start)
    stages = []
    stages.append({'model': 'cv', 'len': 333, 'velocity': [200, 0]})
    stages.append({'model': 'ct', 'len': 333, 'omega': tlb.deg2rad(10)})
    stages.append({'model': 'ca', 'len': 333, 'acceleration': 3})
    traj.add_stage(stages)
    traj.show_traj()
    R = np.eye(2)
    traj_real, traj_meas = traj(R)
    N = len(traj)

    # CV
    cv_xdim, cv_zdim = 4, 2
    sigma_w = [np.sqrt(1.0), np.sqrt(1.0)]
    sigma_v = [np.sqrt(1.0), np.sqrt(1.0)]
    F = model.F_cv(1, T)
    H = model.H_cv(1)
    L = np.eye(cv_xdim)
    M = np.eye(cv_zdim)
    Q = model.Q_cv_dd(1, T, sigma_w)
    R = np.diag(sigma_v)
    cv_kf = ft.KFilter(F, L, H, M, Q, R, cv_xdim, cv_zdim)

    # CA
    ca_xdim, ca_zdim = 6, 2
    sigma_w = [np.sqrt(1.0), np.sqrt(1.0)]
    sigma_w = [np.sqrt(1.0), np.sqrt(1.0)]
    sigma_v = [np.sqrt(1.0), np.sqrt(1.0)]
    F = model.F_ca(1, T)
    H = model.H_ca(1)
    L = np.eye(ca_xdim)
    M = np.eye(ca_zdim)
    Q = model.Q_ca_dd(1, T, sigma_w)
    R = np.diag(sigma_v)
    ca_kf = ft.KFilter(F, L, H, M, Q, R, ca_xdim, ca_zdim)

    # CT
    ct_xdim, ct_zdim = 4, 2
    sigma_w = [np.sqrt(1.0), np.sqrt(1.0)]
    sigma_v = [np.sqrt(1.0), np.sqrt(1.0)]
    turn_rate = tlb.deg2rad(10)
    F = model.F_ct2D(turn_rate, T)
    H = model.H_ct2D(1)
    L = np.eye(ct_xdim)
    M = np.eye(ct_zdim)
    Q = model.Q_ct2D(T, sigma_w)
    R = np.diag(sigma_v)
    ct_kf = ft.KFilter(F, L, H, M, Q, R, ct_xdim, ct_zdim)

    r = 3
    
    dmmf = ft.IMMFilter(xdim, zdim)
    dmmf.add_models([cv_kf, ca_kf, ct_kf], ['cv', 'ca', 'ct2D'])

    x_init = start
    P_init = np.diag([1, 1e3, 1e3, 1, 1e3, 1e3])
    # x_init, P_init = start, np.eye(xdim)
    dmmf.init(x_init, P_init)

    post_state_arr = np.empty((xdim, N))
    prob_arr = np.empty((r, N))

    post_state_arr[:, 0] = dmmf.post_state
    prob_arr[:, 0] = dmmf.probs()
    for n in range(1, N):
        dmmf.step(traj_meas[:, n])

        post_state_arr[:, n] = dmmf.post_state
        prob_arr[:, n] = dmmf.probs()
        print(n)
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
        ax.plot(n, prob_arr[i, :], linewidth=0.8)
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.set_ylim([0, 1])
    ax.legend([str(n) for n in range(r)])
    plt.show()


if __name__ == '__main__':
    # gen_traj()
    DMMF_test()