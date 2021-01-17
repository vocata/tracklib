#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as io
import scipy.stats as st
import tracklib.model as model
import tracklib.utils as utils


def gen_ellipse_uniform(trajs, C, R, theta, lamb, pd=1):
    N = trajs.shape[0]
    trajs_ellip = []
    real_ellip = []
    A = np.eye(2)
    for i in range(N):
        pt_N = st.poisson.rvs(lamb)
        A = utils.rotate_matrix_deg(theta[i]) @ A
        real_ellip.append(A @ C @ A.T)
        z = utils.ellip_uniform(real_ellip[i], pt_N)
        z += st.multivariate_normal.rvs(cov=R, size=pt_N)
        z += trajs[i]
        idx = []
        for j in range(pt_N):
            if np.random.rand() < pd:
                idx.append(j)
        z = z[idx]
        trajs_ellip.append(z)
    return trajs_ellip, real_ellip


def ExtendedObjectFormation():
    vel = 50e3 / 36e2   # 27 knot
    record = {
        'interval': [10],
        'start': [
            [0, 0, 0],
        ],
        'pattern': [
            [
                {'model': 'cv', 'length': 41, 'velocity': [vel * np.sin(np.pi / 4), -vel * np.cos(np.pi / 4), 0]},
                {'model': 'ct', 'length': 9, 'turnrate': 45/90},
                {'model': 'cv', 'length': 27, 'velocity': vel},
                {'model': 'ct', 'length': 15, 'turnrate': 90/150},
                {'model': 'cv', 'length': 18, 'velocity': vel},
                {'model': 'ct', 'length': 15, 'turnrate': 90/150},
                {'model': 'cv', 'length': 54, 'velocity': vel},
            ],
        ],
        'noise': [model.R_cv(3, [0., 0., 0.])],
        'entries': 1
    }
    trajs_state, trajs_meas = model.trajectory_generator(record)
    trajs_meas = model.trajectory_with_pd(trajs_meas, pd=1)
    trajs_state = trajs_state[0]

    C = np.diag([340 / 2, 80 / 2])**2

    axis = 2
    sigma_v = [50, 50]

    R = model.R_cv(axis, sigma_v)

    theta = [-45]
    theta.extend([0] * 41)
    theta.extend([45 / 9 for _ in range(9)])
    theta.extend([0] * 27)
    theta.extend([90 / 15 for _ in range(15)])
    theta.extend([0] * 18)
    theta.extend([90 / 15 for _ in range(15)])
    theta.extend([0] * 54)
    trajs_meas_ellip, real_ellip = gen_ellipse_uniform(trajs_meas[0][:, :-1], C, R, theta, 20)

    return trajs_state, trajs_meas_ellip, real_ellip


if __name__ == '__main__':
    trajs_state, trajs_meas, real_ellip = [], [], []
    for i in range(1000):
        state, meas, ellip = ExtendedObjectFormation()
        trajs_state.append(state)
        trajs_meas.append(meas)
        real_ellip.append(ellip)
    io.savemat('eot_data.mat', {'trajs_state': trajs_state[0], 'trajs_meas': trajs_meas, 'real_ellip': real_ellip[0]})
