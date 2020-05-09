#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pic
import numpy as np
import scipy.io as io
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import tracklib.tracker as tk
import matplotlib.pyplot as plt


def GNNTracker_test():
    # load data
    with open('traj.dat', 'rb') as f:
        traj = pic.load(f)
    true_pos = traj['true_pos']
    meas_pos = traj['meas_pos']
    meas_cov = traj['meas_cov']
    N, T = traj['len'], traj['T']

    axis = 3
    cv_xdim, cv_zdim = 6, 3

    # filter
    sigma_w = [30, 30, 1]     # Increase the filter process noise to account for unknown acceleration.
    sigma_v = np.sqrt(1000)   # measurement noise can be ignored because GNN tracker will reset it later
    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    L = np.eye(cv_xdim)
    M = np.eye(cv_zdim)
    Q = model.Q_cv_dd(axis, T, sigma_w)
    R = model.R_cv(axis, sigma_v)
    ft_gen = tk.GNNFilterGenerator(ft.KFilter, F, L, H, M, Q, R)

    # initializer
    vmax = 1200e3/3600        # 1200km/h, vmax is used to initialize state covariance
    vxmax = vymax = vmax
    vzmax = 10
    ft_init = tk.GNNFilterInitializer(init.cv_init, vmax=[vxmax, vymax, vzmax])

    # logic
    lgc = tk.GNNLogicMaintainer(tk.HistoryLogic, 3, 4, 6, 6)

    # The normalized Mahalanobis distance with penalty term is used,
    # so the threshold is higher than that without penalty term
    threshold = 45

    # initialize the tracker
    tracker = tk.GNNTracker(ft_gen, ft_init, lgc, threshold)

    track_history = {}

    for n in range(N):
        meas = meas_pos[n]
        cov = meas_cov[n]
        det = tk.Detection(meas, cov)

        tracker.add_detection(det)

        tracks = tracker.tracks()
        for track in tracks:
            if track.id not in track_history:
                track_history[track.id] = [track.state]
            else:
                track_history[track.id].append(track.state)
        # print(tracker.current_tracks_num())

    print(tracker.history_tracks_num())

    # plot
    xlim = (-1600, 2000)
    ylim = (-20100, -18100)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(true_pos[0][0, :], true_pos[0][1, :], '-.', color='gray', linewidth=0.6)
    ax.plot(true_pos[1][0, :], true_pos[1][1, :], '-.', color='gray', linewidth=0.6, label='true')
    for i in range(N):
        for j in range(len(meas_pos[i])):
            ax.scatter(meas_pos[i][j][0], meas_pos[i][j][1], s=5, c='orange')
            ax.scatter(meas_pos[i][j][0], meas_pos[i][j][1], s=5, c='orange')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal'), ax.grid(), ax.legend()
    ax.set_title('true trajectory and measurement')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    id = track_history.keys()
    for i in id:
        state = np.array(track_history[i], dtype=float).T
        ax.plot(state[0, :], state[2, :], linewidth=0.8, label='id %d' % i)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal'), ax.grid(), ax.legend()
    ax.set_title('estimation')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()


if __name__ == '__main__':
    GNNTracker_test()