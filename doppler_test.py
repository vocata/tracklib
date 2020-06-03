#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
import tracklib as tlb
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import matplotlib.pyplot as plt
from tracklib import Scope
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
    start = np.array([100, 0, 0, 100, 0, 0, 100, 0, 0], dtype=float)
    sigma_v2 = [5, tlb.deg2rad(0.0001), tlb.deg2rad(0.0001), 0.05]
    traj = model.Trajectory(T,
                            np.diag(sigma_v2),
                            start=start,
                            pd=[(Scope(0, 30), 0.3), (Scope(30, np.inf), 0.8)])
    stages = []
    stages.append({'model': 'cv', 'len': 333, 'vel': [200, 0, 1]})
    stages.append({'model': 'ct', 'len': 333, 'omega': 10})
    stages.append({'model': 'ca', 'len': 333, 'acc': 3})

    traj.add_stage(stages)
    traj_real, traj_meas = traj('rae')
    N = len(traj)

    model_cls = []
    model_types = []
    init_args = []
    init_kwargs = []

    # CV
    cv_xdim, cv_zdim = 6, 4
    sigma_w = np.sqrt(1.0)

    f = model.f_cv(axis, T)
    fjac = model.f_cv_jac(axis, T)
    L = np.eye(cv_xdim)
    Q = model.Q_cv_dd(axis, T, sigma_w)

    def h(x):
        p, v = x[0::2], x[1::2]
        d = lg.norm(p)
        speed = np.dot(p, v) / d
        r, az, elev = tlb.cart2sph(*p)
        return np.array([r, az, elev, speed], dtype=float)
    M = np.eye(cv_zdim)
    R = np.diag(sigma_v2)

    model_cls.append(ft.EKFilterAN)
    model_types.append('cv')
    init_args.append((f, L, h, M, Q, R, cv_xdim, cv_zdim))
    init_kwargs.append({'fjac': fjac})

    # CA
    ca_xdim, ca_zdim = 9, 4
    sigma_w = np.sqrt(1.0)
    
    f = model.f_ca(axis, T)
    fjac = model.f_ca_jac(axis, T)
    L = np.eye(ca_xdim)
    Q = model.Q_ca_dd(axis, T, sigma_w)

    def h(x):
        p, v = x[0::3], x[1::3]
        d = lg.norm(p)
        speed = np.dot(p, v) / d
        r, az, elev = tlb.cart2sph(*p)
        return np.array([r, az, elev, speed], dtype=float)
    M = np.eye(ca_zdim)
    R = np.diag(sigma_v2)

    model_cls.append(ft.EKFilterAN)
    model_types.append('ca')
    init_args.append((f, L, h, M, Q, R, ca_xdim, ca_zdim))
    init_kwargs.append({'fjac': fjac})

    # CT
    ct_xdim, ct_zdim = 7, 4
    sigma_w = np.sqrt(1.0)

    f = model.f_ct(axis, T)
    fjac = model.f_ct_jac(axis, T)
    L = np.eye(ct_xdim)
    Q = model.Q_ct(axis, T, sigma_w)

    def h(x):
        p, v = x[[0, 2, 5]], x[[1, 3, 6]]
        d = lg.norm(p)
        speed = np.dot(p, v) / d
        r, az, elev = tlb.cart2sph(*p)
        return np.array([r, az, elev, speed], dtype=float)
    M = np.eye(ct_zdim)
    R = np.diag(sigma_v2)

    model_cls.append(ft.EKFilterAN)
    model_types.append('ct')
    init_args.append((f, L, h, M, Q, R, ct_xdim, ct_zdim))
    init_kwargs.append({'fjac': fjac})

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

    x_init = np.array([100, 0, 100, 0, 100, 0], dtype=float)
    P_init = np.diag([1.0, 1e4, 1.0, 1e4, 1.0, 1e4])
    dmmf.init(x_init, P_init)

    post_state_arr = np.empty((cv_xdim, N))
    prob_arr = np.empty((r, N))

    post_state_arr[:, 0] = dmmf.state
    prob_arr[:, 0] = dmmf.probs()
    for n in range(1, N):
        dmmf.predict()
        z = traj_meas[:, n]
        if not np.any(np.isnan(z)):     # skip the empty detections
            dmmf.correct(z)

        post_state_arr[:, n] = dmmf.state
        prob_arr[:, n] = dmmf.probs()

    print(dmmf)

    # trajectory
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(traj_real[0, 0], traj_real[1, 0], traj_real[2, 0], s=50, c='r', marker='x', label='start')
    ax.plot(traj_real[0, :], traj_real[1, :], traj_real[2, :], linewidth=0.8, label='real')
    ax.scatter(*tlb.sph2cart(*traj_meas[:3]), s=5, c='orange', label='meas')
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
