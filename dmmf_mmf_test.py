#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tracklib as tlb
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import matplotlib.pyplot as plt
from tracklib import Scope, Pair
from mpl_toolkits import mplot3d
'''
notes:
vector is preferably a column vector, otherwise
the program may yield uncertain result.
'''


def DMMF_MMF_test():
    T = 0.1
    axis = 3

    # generate trajectory
    start = np.array([100, 0, 0, 100, 0, 0, 100, 0, 0], dtype=float)
    traj = model.Trajectory(T,
                            np.eye(axis),
                            start=start,
                            pd=[(Scope(0, 30), 0.3), (Scope(30, np.inf), 0.8)])
    stages = []
    stages.append({'model': 'cv', 'len': 200, 'vel': [150, 0, 0]})
    stages.append({'model': 'ct', 'len': 200, 'omega': -8})
    stages.append({'model': 'ca', 'len': 200, 'acc': [None, None, 3]})
    stages.append({'model': 'ct', 'len': 200, 'omega': 5})
    stages.append({'model': 'cv', 'len': 200, 'vel': 50})
    stages.append({'model': 'ca', 'len': 200, 'acc': 3})
    traj.add_stage(stages)
    traj.show_traj()
    traj_real, traj_meas = traj()
    N = len(traj)

    model_cls1 = []
    model_types1 = []
    init_args1 = []
    init_kwargs1 = []

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
    model_cls1.append(ft.KFilter)
    model_types1.append('cv')
    init_args1.append((F, L, H, M, Q, R))
    init_kwargs1.append({})

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
    model_cls1.append(ft.KFilter)
    model_types1.append('ca')
    init_args1.append((F, L, H, M, Q, R))
    init_kwargs1.append({})

    model_cls2 = []
    model_types2 = []
    init_args2 = []
    init_kwargs2 = []

    model_cls2.append(ft.MMFilter)
    model_types2.append('cv')
    init_args2.append((model_cls1, model_types1, init_args1, init_kwargs1))
    init_kwargs2.append({})

    # CT
    ct_xdim, ct_zdim = 7, 3
    sigma_w = np.sqrt(1.0)
    sigma_v = np.sqrt(1.0)
    f = model.f_ct(axis, T)
    fjac = model.f_ct_jac(axis, T)
    L = np.eye(ct_xdim)
    h = model.h_ct(axis)
    hjac = model.h_ct_jac(axis)
    M = np.eye(ct_zdim)
    Q = model.Q_ct(axis, T, sigma_w)
    R = model.R_ct(axis, sigma_v)
    model_cls2.append(ft.EKFilterAN)
    model_types2.append('ct')
    init_args2.append((f, L, h, M, Q, R, ct_xdim, ct_zdim))
    init_kwargs2.append({'fjac': fjac, 'hjac': hjac})

    # number of models
    r = 2

    dmmf = ft.IMMFilter(model_cls2, model_types2, init_args2, init_kwargs2)

    x_init = np.array([100, 0, 100, 0, 100, 0], dtype=float)
    P_init = np.diag([1.0, 1e4, 1.0, 1e4, 1.0, 1e4])
    dmmf.init(x_init, P_init)

    post_state_arr = np.empty((cv_xdim, N))
    dmmf_prob_arr = np.empty((r, N))

    post_state_arr[:, 0] = dmmf.state
    dmmf_prob_arr[:, 0] = dmmf.probs()
    for n in range(1, N):
        dmmf.predict()
        z = traj_meas[:, n]
        if not np.any(np.isnan(z)):
            dmmf.correct(z)

        post_state_arr[:, n] = dmmf.state
        dmmf_prob_arr[:, n] = dmmf.probs()

    print(dmmf)

    # trajectory
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
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
    labels = ['hybrid', 'ct']
    for i in range(r):
        ax.plot(n, dmmf_prob_arr[i, :], linewidth=0.8, label=labels[i])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.set_xlim([0, 1200])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.set_title('models probability')
    plt.show()


if __name__ == '__main__':
    DMMF_MMF_test()