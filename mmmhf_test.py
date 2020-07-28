#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import tracklib.filter as ft
import tracklib.model as model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
'''
notes:
vector is preferably a column vector, otherwise
the program may yield uncertain result.
'''


def MMMHF_test():
    T = 0.1
    axis = 3
    
    # generate trajectory
    record = {
        'interval': [T],
        'start': [[100, 100, 100]],
        'pattern': [
            [
                {'model': 'cv', 'length': 333, 'velocity': [200, 0, 1]},
                {'model': 'ct', 'length': 333, 'turnrate': 10},
                {'model': 'ca', 'length': 333, 'acceleration': 3}
            ]
        ],
        'noise':[np.eye(axis)],
        'pd': [0.9],
        'entries': 1
    }
    trajs_state, trajs_meas = model.trajectory_generator(record)
    traj_state, traj_meas = trajs_state[0], trajs_meas[0]
    N = traj_state.shape[0]

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
    # init_kwargs.append({'fjac': fjac, 'hjac': hjac, 'order': 2})

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

    mmmhf = ft.MMMHFilter(model_cls, model_types, init_args, init_kwargs, depth=1, pruning=0.0, trans_mat=0.99)

    x_init = np.array([100, 0, 100, 0, 100, 0], dtype=float)
    P_init = np.diag([1.0, 1e4, 1.0, 1e4, 1.0, 1e4])
    mmmhf.init(x_init, P_init)

    post_state_arr = np.empty((N, cv_xdim))

    post_state_arr[0, :] = mmmhf.state

    start = time.time()
    for n in range(1, N):
        mmmhf.predict(n=n)
        z = traj_meas[n, :]
        if not np.any(np.isnan(z)):     # skip the empty detections
            mmmhf.correct(z)

        post_state_arr[n, :] = mmmhf.state
    end = time.time()

    print(mmmhf, 'time: {}'.format(end - start), sep='\n')

    real_state = np.delete(traj_state, np.s_[2::3], axis=1)
    state_err = real_state - post_state_arr
    print('RMS: %s' % np.std(state_err, axis=0))

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


if __name__ == '__main__':
    MMMHF_test()
