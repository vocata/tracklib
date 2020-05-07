#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as io
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import tracklib.tracker as tk
import matplotlib.pyplot as plt


def GNNTracker_test():
    mat = io.loadmat(r'C:\Users\Ray\Desktop\MTT\matlab.mat')
    truePos = mat['truePos'][0].tolist()
    measPos = mat['measPos'][0].tolist()
    measCov = mat['measCov'][0].tolist()
    time = mat['time'][0]
    N = len(time)
    T = np.mean(np.diff(time))

    speed = 1e6/3600
    axis = 3
    cv_xdim, cv_zdim = 6, 3
    sigma_w = np.sqrt(1000)
    sigma_v = np.sqrt(1000)
    F = model.F_cv(axis, T)
    H = model.H_cv(axis)
    L = np.eye(cv_xdim)
    M = np.eye(cv_zdim)
    Q = model.Q_cv_dd(axis, T, sigma_w)
    R = model.R_cv(axis, sigma_v)
    ft_gen = tk.GNNFilterGenerator(ft.KFilter, F, L, H, M, Q, R)

    ft_init = tk.GNNFilterInitializer(init.cv_init, v_max=speed)

    lgc = tk.GNNLogicMaintainer(tk.HistoryLogic, 2, 3, 6, 6)

    tracker = tk.GNNTracker(ft_gen, ft_init, lgc, 45)

    tracks = {}

    for n in range(N):
        meas_num = measPos[n].shape[0]
        meas = [measPos[n][i, :] for i in range(meas_num)]
        meas_cov = [measCov[n][:, :, i] if meas_num > 1 else measCov[n] for i in range(meas_num)]
        det = tk.Detection(meas, meas_cov)

        tracker.add_detection(det)

        all_tracks = tracker.all_tracks()
        for t in all_tracks:
            if t.id not in tracks:
                tracks[t.id] = [t.state]
            else:
                tracks[t.id].append(t.state)

    print(tracker.history_tracks_num())

    fig = plt.figure()
    ax = fig.add_subplot()
    t_id = tracks.keys()
    for id in t_id:
        state = np.array(tracks[id], dtype=float)
        ax.plot(state[:, 0], state[:, 2])
    plt.show()
        

if __name__ == '__main__':
    GNNTracker_test()