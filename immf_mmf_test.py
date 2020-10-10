#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tracklib.filter as ft
import tracklib.model as model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def DMMF_MMF_test():
    T = 0.1
    axis = 3

    # generate trajectory
    record = {
        'interval': [T],
        'start': [[100, 100, 100]],
        'pattern': [
            [
                {'model': 'cv', 'length': 200, 'velocity': [150, 0, 0]},
                {'model': 'ct', 'length': 200, 'turnrate': -8},
                {'model': 'ca', 'length': 200, 'acceleration': [None, None, 3]},
                {'model': 'ct', 'length': 200, 'turnrate': 5},
                {'model': 'cv', 'length': 200, 'velocity': 50},
                {'model': 'ca', 'length': 200, 'acceleration': 3}
            ]
        ],
        'noise':[np.eye(axis)],
        'pd': [0.9],
        'entries': 1
    }
    trajs_state, trajs_meas = model.trajectory_generator(record)
    traj_state, traj_meas = trajs_state[0], trajs_meas[0]
    N = traj_state.shape[0]

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

    immf = ft.IMMFilter(model_cls2, model_types2, init_args2, init_kwargs2)

    x_init = np.array([100, 0, 100, 0, 100, 0], dtype=float)
    P_init = np.diag([1.0, 1e4, 1.0, 1e4, 1.0, 1e4])
    immf.init(x_init, P_init)

    post_state_arr = np.empty((N, cv_xdim))
    immf_prob_arr = np.empty((N, r))

    post_state_arr[0, :] = immf.state
    immf_prob_arr[0, :] = immf.probs()
    for n in range(1, N):
        immf.predict()
        z = traj_meas[n, :]
        if not np.any(np.isnan(z)):
            immf.correct(z)

        post_state_arr[n, :] = immf.state
        immf_prob_arr[n, :] = immf.probs()

    print(immf)

    # trajectory
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(traj_state[0, 0], traj_state[0, 3], traj_state[0, 6], s=50, c='r', marker='x', label='start')
    ax.plot(traj_state[:, 0], traj_state[:, 3], traj_state[:, 6], linewidth=0.8, label='real')
    ax.scatter(traj_meas[:, 0], traj_meas[:, 1], traj_meas[:, 2], s=5, c='orange', label='meas')
    ax.plot(post_state_arr[:, 0], post_state_arr[:, 2], post_state_arr[:, 4], linewidth=0.8, label='esti')
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
        ax.plot(n, immf_prob_arr[:, i], linewidth=0.8, label=labels[i])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.set_xlim([0, 1200])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.set_title('models probability')
    plt.show()


if __name__ == '__main__':
    DMMF_MMF_test()