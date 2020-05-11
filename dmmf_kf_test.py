#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
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


def DMMF_test():
    T = 0.1
    axis = 3

    # generate trajectory
    np.random.seed(2020)
    start = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    traj = model.Trajectory(T, start=start)
    stages = []
    stages.append({'model': 'cv', 'len': 333, 'vel': [200, 0, 1]})
    stages.append({'model': 'ct', 'len': 333, 'omega': 10})
    stages.append({'model': 'ca', 'len': 333, 'acc': 3})

    # stages.append({'model': 'cv', 'len': 200, 'vel': [150, 0, 0]})
    # stages.append({'model': 'ct', 'len': 200, 'omega': -8})
    # stages.append({'model': 'ca', 'len': 200, 'acc': [None, None, 3]})
    # stages.append({'model': 'ct', 'len': 200, 'omega': 5})
    # stages.append({'model': 'cv', 'len': 200, 'vel': 50})
    # stages.append({'model': 'ca', 'len': 200, 'acc': 3})

    R = np.eye(3)
    traj.add_stage(stages, R)
    traj.show_traj()
    traj_real, traj_meas = traj()
    N = len(traj)

    # traj_real = np.loadtxt(
    #     r'C:\Users\Ray\Documents\MATLAB\Examples\R2020a\fusion\TrackingManeuveringTargetsExample\truePos.csv',
    #     dtype=np.float64,
    #     delimiter=',')
    # traj_meas = np.loadtxt(
    #     r'C:\Users\Ray\Documents\MATLAB\Examples\R2020a\fusion\TrackingManeuveringTargetsExample\measPos.csv',
    #     dtype=np.float64,
    #     delimiter=',')
    # N = traj_meas.shape[1]

    model_cls = []
    model_types = []
    init_args = []
    init_kwargs = []

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
    model_cls.append(ft.KFilter)
    model_types.append('cv')
    init_args.append((F, L, H, M, Q, R))
    init_kwargs.append({})

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
    model_cls.append(ft.KFilter)
    model_types.append('ca')
    init_args.append((F, L, H, M, Q, R))
    init_kwargs.append({})

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
    model_cls.append(ft.EKFilterAN)
    model_types.append('ct')
    init_args.append((f, L, h, M, Q, R, ct_xdim, ct_zdim))
    init_kwargs.append({'fjac': fjac, 'hjac': hjac})

    # pt_gen = ft.ScaledSigmaPoints()
    # model_cls.append(ft.UKFilterAN)
    # model_types.append('ct')
    # init_args.append((f, L, h, M, Q, R, pt_gen))
    # init_kwargs.append({})

    # model_cls.append(ft.SIRPFilter)
    # model_types.append('ct')
    # init_args.append((f, L, h, M, Q, R, 200, 100))
    # init_kwargs.append({})

    # kernal = ft.EpanechnikovKernal(ct_xdim, 200)
    # kernal = ft.GaussianKernal(ct_xdim, 200)
    # model_cls.append(ft.RPFilter)
    # model_types.append('ct')
    # init_args.append((f, L, h, M, Q, R, 200, 100))
    # init_kwargs.append({'kernal': kernal})

    # model_cls.append(ft.GPFilter)
    # model_types.append('ct')
    # init_args.append((f, L, h, M, Q, R, 200))
    # init_kwargs.append({})

    # number of models
    r = 3

    dmmf = ft.IMMFilter(model_cls, model_types, init_args, init_kwargs)

    x_init = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    P_init = np.diag([1.0, 1e4, 1.0, 1e4, 1.0, 1e4])
    dmmf.init(x_init, P_init)

    post_state_arr = np.empty((cv_xdim, N))
    prob_arr = np.empty((r, N))

    post_state_arr[:, 0] = dmmf.state
    prob_arr[:, 0] = dmmf.probs()
    for n in range(1, N):
        dmmf.predict()
        dmmf.correct(traj_meas[:, n])

        post_state_arr[:, n] = dmmf.state
        prob_arr[:, n] = dmmf.probs()

    print(dmmf)

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
        ax.plot(n, prob_arr[i, :], linewidth=0.8, label=model_types[i])
    ax.set_xlabel('time(s)')
    ax.set_ylabel('probability')
    ax.set_xlim([0, 1200])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.set_title('models probability')
    plt.show()


if __name__ == '__main__':
    DMMF_test()