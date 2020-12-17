import os
import imageio
import numpy as np
import scipy.io as io
import tracklib.filter as ft
import tracklib.init as init
import tracklib.model as model
import tracklib.tracker as tk
import matplotlib.pyplot as plt


def png_to_fig():
    images = []
    for i in range(len(os.listdir('data'))):
        images.append(imageio.imread('data/fig%d.png' % i))
    imageio.mimsave('data/merge.gif', images, format='GIF', duration=0.1)


def GNN_doppler(save=False):
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
    sigma_w = [1, 1, 1]   # [x, y, z]
    sigma_v = [5, 0.005, 0.005]           # [r, az, elev]

    F = model.F_cv(axis, T)
    L = np.eye(cv_xdim)
    Q = model.Q_cv_dd(axis, T, sigma_w)

    H = model.H_cv(axis)
    M = np.eye(cv_zdim)
    R = np.diag(sigma_v)**2

    # use CMKF filter generator
    ft_gen = tk.GNNFilterGenerator(ft.KFilter, F, L, H, M, Q, R)

    # filter initializer
    vxmax, vymax, vzmax = 5e3/3600, 5e3/3600, 1e3/3600
    ft_init = tk.GNNFilterInitializer(init.cv_init, vmax=[vxmax, vymax, vzmax])

    # logic
    lgc = tk.GNNLogicMaintainer(tk.HistoryLogic, 4, 4, 6, 6)

    # initialize the tracker
    tracker = tk.GNNTracker(ft_gen, ft_init, lgc, 35)

    # variable recording the tracking
    state_history = {}
    line_history = {}
    clutters = None

    # figure setting
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.set_xlim([0, 2500])
    ax.set_ylim([-1200, 1200])
    fig_m = plt.get_current_fig_manager()
    fig_m.window.showMaximized()

    plt.ion()
    for n in range(N):
        # form the detection
        meas, cov = [], []
        for point in pt[n]:
            z = point[:3]
            z_cart, R_cart = model.convert_meas(z, R, elev=True)
            meas.append(z_cart)
            cov.append(R_cart)
        det = tk.Detection(meas, cov)

        # plot all points
        meas_arr = np.reshape(meas, (-1, 3))
        if not clutters:
            clutters, = ax.plot(meas_arr[:, 0], meas_arr[:, 1], 'k.', markersize=3)
        else:
            x, y = clutters.get_data()
            x = np.concatenate((x, meas_arr[:, 0]))
            y = np.concatenate((y, meas_arr[:, 1]))
            clutters.set_data(x, y)

        # tracking iteration
        tracker.add_detection(det)
        tracks = tracker.tracks()

        # plot tracks
        ax.texts = []
        for track in tracks:
            if track.id not in state_history:
                state_history[track.id] = track.state.reshape(1, -1)
                x, y = state_history[track.id][:, 0], state_history[track.id][:, 2]
                line_history[track.id] = ax.plot(x, y, 'g', x[-1], y[-1], '.', markersize=10)
            else:
                state_history[track.id] = np.vstack((state_history[track.id], track.state))
                x, y = state_history[track.id][:, 0], state_history[track.id][:, 2]
                line_history[track.id][0].set_data(x, y)
                line_history[track.id][1].set_data(x[-1], y[-1])
            ax.text(track.state[0], track.state[2], 'T%d, age:%d' % (track.id, track.age))

        ax.set_title('trajectory, num: %d' % tracker.history_tracks_num())
        plt.draw_if_interactive()
        plt.pause(0.01)
        if save:
            plt.savefig('data/fig%d.png' % n)
    plt.ioff()
    plt.show()
    if save:
        png_to_fig()


if __name__ == '__main__':
    save = False
    GNN_doppler(save)
