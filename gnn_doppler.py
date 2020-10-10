import numpy as np
import scipy.io as io
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import tracklib.tracker as tk
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def GNN_doppler():
    # load data
    data = io.loadmat('pt_py.mat')
    dat_len = data['pt'].shape[1]
    pt = [[vec for vec in data['pt'][0, i]] for i in range(dat_len)]


    # simulation duration and scan interval
    N, T = dat_len, 0.82

    # spatial dimension of target motion
    axis = 3

    # CV
    cv_xdim, cv_zdim = 6, 3
    sigma_w = [1, 1, 1]   # [x, y, y]
    sigma_v = [10, 0.01, 0.01]           # [r, az, elev]

    F = model.F_cv(axis, T)
    L = np.eye(cv_xdim)
    Q = model.Q_cv_dd(axis, T, sigma_w)

    H = model.H_cv(axis)
    M = np.eye(cv_zdim)
    R = np.diag(sigma_v)**2

    # use CMKF filter generator
    ft_gen = tk.GNNFilterGenerator(ft.KFilter, F, L, H, M, Q, R)

    # filter initializer
    vxmax, vymax, vzmax = 80e3/3600, 80e3/3600, 10e3/3600
    ft_init = tk.GNNFilterInitializer(init.cv_init, vmax=[vxmax, vymax, vzmax])

    # logic
    lgc = tk.GNNLogicMaintainer(tk.HistoryLogic, 3, 4, 5, 5)

    # initialize the tracker
    tracker = tk.GNNTracker(ft_gen, ft_init, lgc, 25)

    # tracking process
    state_history = {}
    for n in range(N):
        meas, cov = [], []
        for i in range(len(pt[n])):
            z = pt[n][i][:3]
            z_cart, R_cart = model.convert_meas(z, R, elev=True)
            meas.append(z_cart)
            cov.append(R_cart)
        det = tk.Detection(meas, cov)

        tracker.add_detection(det)

        tracks = tracker.tracks()
        for track in tracks:
            if track.id not in state_history:
                state_history[track.id] = [track.state]
            else:
                state_history[track.id].append(track.state)
        print(tracker.current_tracks_num())

    print('total number of tracks: %d' % tracker.history_tracks_num())

    # find longest track and plot it
    esti_state = sorted(state_history.values(), key=lambda x: len(x), reverse=True)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(10):
        ax.plot(np.array(esti_state[i])[:, 0], np.array(esti_state[i])[:, 2], np.array(esti_state[i])[:, 4])
    plt.show()


if __name__ == '__main__':
    GNN_doppler()