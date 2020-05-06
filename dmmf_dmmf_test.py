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


def DMMF_DMMF_test():
    T = 0.1
    axis = 3

    # generate trajectory
    start = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    traj = model.Trajectory2D(T, start)
    stages = []
    stages.append({'model': 'cv', 'len': 200, 'vel': [150, 0, 0]})
    stages.append({'model': 'ct', 'len': 200, 'omega': -8})
    stages.append({'model': 'ca', 'len': 200, 'acc': [None, None, 3]})
    stages.append({'model': 'ct', 'len': 200, 'omega': 5})
    stages.append({'model': 'cv', 'len': 200, 'vel': 50})
    stages.append({'model': 'ca', 'len': 200, 'acc': 3})
    traj.add_stage(stages)
    traj.show_traj()
    R = np.eye(3)
    traj_real, traj_meas = traj(R)
    N = len(traj)

    # CV
    cv_xdim, cv_zdim = 6, 3
    sigma_w = np.sqrt(1.0)
    sigma_v = np.sqrt(1.0)
    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    L = np.eye(cv_xdim)
    M = np.eye(cv_zdim)
    Q = model.Q_cv_dd(axis, T, sigma_w)
    R = model.R_cv(axis, sigma_v)
    cv_kf = ft.KFilter(F, L, H, M, Q, R)

    # CA
    ca_xdim, ca_zdim = 9, 3
    sigma_w = np.sqrt(1.0)
    sigma_v = np.sqrt(1.0)
    F = model.F_ca(axis, T)
    H = model.H_ca(axis)
    L = np.eye(ca_xdim)
    M = np.eye(ca_zdim)
    Q = model.Q_ca_dd(axis, T, sigma_w)
    R = model.R_ca(axis, sigma_v)
    ca_kf = ft.KFilter(F, L, H, M, Q, R)
    # mmf including CV and CA
    dmmf_models1 = [cv_kf, ca_kf]
    mmf_types1 = ['cv', 'ca']
    dmmf1 = ft.IMMFilter()
    dmmf1.add_models(dmmf_models1, mmf_types1)

    # CT
    ct_xdim, ct_zdim = 7, 3
    sigma_w = np.sqrt(1.0)
    sigma_v = np.sqrt(1.0)
    f = model.f_ct2D(axis, T)
    fjac = model.f_ct2D_jac(axis, T)
    L = np.eye(ct_xdim)
    h = model.h_ct2D(axis)
    hjac = model.h_ct2D_jac(axis)
    M = np.eye(ct_zdim)
    Q = model.Q_ct2D(axis, T, sigma_w)
    R = model.R_ct2D(axis, sigma_v)
    ct_ekf = ft.EKFilterAN(f, L, h, M, Q, R, ct_xdim, ct_zdim, fjac=fjac, hjac=hjac)

    # number of models
    r = 2

    dmmf_models2 = [dmmf1, ct_ekf]
    dmmf_types2 = ['cv', 'ct2D']
    dmmf2 = ft.IMMFilter()
    dmmf2.add_models(dmmf_models2, dmmf_types2)

    x_init = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    P_init = np.diag([1.0, 1e4, 1.0, 1e4, 1.0, 1e4])
    dmmf2.init(x_init, P_init)

    post_state_arr = np.empty((cv_xdim, N))
    dmmf2_prob_arr = np.empty((r, N))
    dmmf1_prob_arr = np.empty((r, N))

    post_state_arr[:, 0] = dmmf2.state
    dmmf2_prob_arr[:, 0] = dmmf2.probs()
    dmmf1_prob_arr[:, 0] = dmmf1.probs()
    for n in range(1, N):
        dmmf2.predict()
        dmmf2.correct(traj_meas[:, n])

        post_state_arr[:, n] = dmmf2.state
        dmmf2_prob_arr[:, n] = dmmf2.probs()
        dmmf1_prob_arr[:, n] = dmmf1.probs()

    print(dmmf2)

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
    labels = ['hybrid', 'ct2D']
    for i in range(r):
        ax.plot(n, dmmf2_prob_arr[i, :], linewidth=0.8, label=labels[i])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.set_xlim([0, 1200])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.set_title('models probability(dmmf)')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    n = np.arange(N)
    labels = ['cv', 'ca']
    for i in range(r):
        ax.plot(n, dmmf1_prob_arr[i, :], linewidth=0.8, label=labels[i])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.set_xlim([0, 1200])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.set_title('models probability(mmf)')
    plt.show()


if __name__ == '__main__':
    DMMF_DMMF_test()