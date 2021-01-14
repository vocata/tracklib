import numpy as np
import scipy.io as io
import scipy.stats as st
import tracklib.model as model
import tracklib.utils as utils
import matplotlib.pyplot as plt


def plot_ellipse(ax, x0, y0, C, N, *args, **kwargs):
    x, y = utils.ellip_point(x0, y0, C, N)
    ax.plot(x, y, *args, **kwargs)


def GroupFormation():
    record = {
        'interval': [10] * 5,
        'start': [
            [1000, 20000, 0],
            [1000, 20500, 0],
            [1000, 21000, 0],
            [1000, 21500, 0],
            [1000, 22000, 0],
        ],
        'pattern': [
            [
                {'model': 'cv', 'length': 29, 'velocity': [300, 0, 0]},
            ],
            [
                {'model': 'cv', 'length': 29, 'velocity': [300, 0, 0]},
            ],
            [
                {'model': 'cv', 'length': 29, 'velocity': [300, 0, 0]},
            ],
            [
                {'model': 'cv', 'length': 29, 'velocity': [300, 0, 0]},
            ],
            [
                {'model': 'cv', 'length': 29, 'velocity': [300, 0, 0]},
            ],
        ],
        'noise': [model.R_cv(3, [0., 0., 0.])] * 5,
        'entries': 5
    }
    trajs_state, real_trajs = model.trajectory_generator(record)
    real_trajs = model.trajectory_with_pd(real_trajs, pd=1)

    for i in range(5):
        real_trajs[i] = real_trajs[i][:, :-1]

    for i in range(5):
        origin = real_trajs[2][10 + i]
        A = utils.rotate_matrix_deg(-18 * (i + 1))
        for j in range(5):
            real_trajs[j][10 + i] = origin + (A @ (real_trajs[j][10 + i] - origin))
    for i in range(15):
        origin = real_trajs[2][15 + i]
        A = utils.rotate_matrix_deg(-90)
        for j in range(5):
            real_trajs[j][15 + i] = origin + (A @ (real_trajs[j][15 + i] - origin))

    R = model.R_cv(3, [500, 100])
    pd = 0.8
    N = trajs_state[2].shape[0]
    # print(N)

    trajs_meas = []
    # add noise
    for i in range(5):
        noi = st.multivariate_normal.rvs(cov=R, size=N)
        trajs_meas.append(real_trajs[i] + noi)

    # remove some measurements according to `pd`
    trajs_meas_remove = []
    for i in range(N):
        remove = st.uniform.rvs(size=5) >= pd
        while remove.sum() == 5:
            remove = st.uniform.rvs(size=5) >= pd
        tmp = []
        for j in range(5):
            if not remove[j]:
                tmp.append(trajs_meas[j][i])
        trajs_meas_remove.append(np.array(tmp, dtype=float))

    return real_trajs, trajs_meas_remove

    # trajectory
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(5):
        ax.scatter(trajs_meas[i][::1, 0], trajs_meas[i][::1, 1], marker='^', facecolors=None, edgecolors='k', s=8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.set_title('trajectory')
    plt.show()


if __name__ == '__main__':
    real_trajs, trajs_meas = [], []
    for i in range(500):
        trajs, meas = GroupFormation()
        real_trajs.append(trajs)
        trajs_meas.append(meas)
    io.savemat('gtt_data.mat', {'real_trajs': real_trajs[0], 'trajs_meas': trajs_meas})