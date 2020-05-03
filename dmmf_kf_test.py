#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tracklib as tlb
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
'''
notes:
vector is preferably a column vector, otherwise
the program may yield uncertain result.
'''


def DMMF_test():
    T = 0.1
    xdim, zdim = 6, 3
    axis = 3

    # generate trajectory
    np.random.seed(2018)
    start = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    traj = model.Trajectory2D(T, start)
    stages = []
    stages.append({'model': 'cv', 'len': 333, 'vel': [200, 0, 1]})
    stages.append({'model': 'ct', 'len': 333, 'omega': 10})
    stages.append({'model': 'ca', 'len': 333, 'acc': 3})
    traj.add_stage(stages)
    traj.show_traj()
    R = np.eye(3)
    traj_real, traj_meas = traj(R)
    N = len(traj)

    # traj_meas = np.loadtxt(
    #     r'C:\Users\Ray\Documents\MATLAB\Examples\R2020a\fusion\TrackingManeuveringTargetsExample\measPos.csv',
    #     dtype=np.float64,
    #     delimiter=',')
    # N = traj_meas.shape[1]

    # CV
    cv_xdim = 6
    sigma_w = [np.sqrt(1.0), np.sqrt(1.0), np.sqrt(1.0)]
    sigma_v = [np.sqrt(1.0), np.sqrt(1.0), np.sqrt(1.0)]
    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    L = np.eye(cv_xdim)
    M = np.eye(zdim)
    Q = model.Q_cv_dd(axis, T, sigma_w)
    R = np.diag(sigma_v)
    cv_kf = ft.KFilter(F, L, H, M, Q, R, cv_xdim, zdim)

    # CA
    ca_xdim = 9
    sigma_w = [np.sqrt(1.0), np.sqrt(1.0), np.sqrt(1.0)]
    sigma_v = [np.sqrt(1.0), np.sqrt(1.0), np.sqrt(1.0)]
    F = model.F_ca(axis, T)
    H = model.H_ca(axis)
    L = np.eye(ca_xdim)
    M = np.eye(zdim)
    Q = model.Q_ca_dd(axis, T, sigma_w)
    R = np.diag(sigma_v)
    ca_kf = ft.KFilter(F, L, H, M, Q, R, ca_xdim, zdim)

    # CT
    ct_xdim = 7
    sigma_w = [np.sqrt(1.0), np.sqrt(1.0), np.sqrt(1.0), np.sqrt(1.0)]
    sigma_v = [np.sqrt(1.0), np.sqrt(1.0), np.sqrt(1.0)]
    f = model.f_ct2D(axis, T)
    fjac = model.f_ct2D_jac(axis, T)
    L = np.eye(ct_xdim)
    h = model.h_ct2D(axis)
    hjac = model.h_ct2D_jac(axis)
    M = np.eye(zdim)
    Q = model.Q_ct2D(axis, T, sigma_w)
    R = np.diag(sigma_v)
    ct_ekf = ft.EKFilterAN(f, L, h, M, Q, R, ct_xdim, zdim, fjac=fjac, hjac=hjac)

    r = 3

    dmmf = ft.IMMFilter(xdim, zdim)
    dmmf.add_models([cv_kf, ca_kf, ct_ekf], ['cv', 'ca', 'ct2D'])

    x_init = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    P_init = np.diag([1.0, 1e4, 1.0, 1e4, 1.0, 1e4])
    dmmf.init(x_init, P_init)

    post_state_arr = np.empty((xdim, N))
    prob_arr = np.empty((r, N))

    post_state_arr[:, 0] = dmmf.post_state
    prob_arr[:, 0] = dmmf.probs()
    for n in range(1, N):
        dmmf.step(traj_meas[:, n])

        post_state_arr[:, n] = dmmf.post_state
        prob_arr[:, n] = dmmf.probs()
    print(len(dmmf))
    print(dmmf)
    print(dmmf.prior_state)
    print(dmmf.post_state)

    # trajectory
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.axis('equal')
    ax.scatter(traj_real[0, 0], traj_real[1, 0], traj_real[2, 0], s=50, c='r', marker='x', label='start')
    ax.plot(traj_real[0, :], traj_real[1, :], traj_real[2, :], linewidth=0.8, label='real')
    ax.scatter(traj_meas[0, :], traj_meas[1, :], traj_meas[2, :], s=5, c='orange', label='meas')
    ax.plot(post_state_arr[0, :], post_state_arr[2, :], post_state_arr[4, :], linewidth=0.8, label='esti')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('trajectory')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    n = np.arange(N)
    for i in range(r):
        ax.plot(n, prob_arr[i, :], linewidth=0.8)
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.set_xlim([0, 1200])
    ax.set_ylim([0, 1])
    ax.legend([str(n) for n in range(r)])
    plt.show()


if __name__ == '__main__':
    # gen_traj()
    DMMF_test()