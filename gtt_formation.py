import time
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
        'interval': [5] * 5,
        'start': [
            [0, 0, 0],
            [0, 500, 0],
            [0, 1000, 0],
            [0, 1500, 0],
            [0, 2000, 0],
        ],
        'pattern': [
            [
                {'model': 'cv', 'length': 30, 'velocity': [300, 0, 0]},
                {'model': 'ct', 'length': 40, 'turnrate': 360/200, 'velocity': 300+2000*np.pi/100},
                {'model': 'cv', 'length': 30, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 90/20, 'velocity': 300+2000*np.pi/40},
                {'model': 'cv', 'length': 30, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 90/20, 'velocity': 300+2000*np.pi/40},
                {'model': 'cv', 'length': 40, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': -10/20},
                {'model': 'cv', 'length': 20, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 30, 'velocity': [300, 0, 0]},
                {'model': 'ct', 'length': 40, 'turnrate': 360/200, 'velocity': 300+1500*np.pi/100},
                {'model': 'cv', 'length': 30, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 90/20, 'velocity': 300+1500*np.pi/40},
                {'model': 'cv', 'length': 30, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 90/20, 'velocity': 300+1500*np.pi/40},
                {'model': 'cv', 'length': 40, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': -5/20},
                {'model': 'cv', 'length': 20, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 30, 'velocity': [300, 0, 0]},
                {'model': 'ct', 'length': 40, 'turnrate': 360/200, 'velocity': 300+1000*np.pi/100},
                {'model': 'cv', 'length': 30, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 90/20, 'velocity': 300+1000*np.pi/40},
                {'model': 'cv', 'length': 30, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 90/20, 'velocity': 300+1000*np.pi/40},
                {'model': 'cv', 'length': 40, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 0/20},
                {'model': 'cv', 'length': 20, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 30, 'velocity': [300, 0, 0]},
                {'model': 'ct', 'length': 40, 'turnrate': 360/200, 'velocity': 300+500*np.pi/100},
                {'model': 'cv', 'length': 30, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 90/20, 'velocity': 300+500*np.pi/40},
                {'model': 'cv', 'length': 30, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 90/20, 'velocity': 300+500*np.pi/40},
                {'model': 'cv', 'length': 40, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 5/20},
                {'model': 'cv', 'length': 20, 'velocity': 300},
            ],
            [
                {'model': 'cv', 'length': 30, 'velocity': [300, 0, 0]},
                {'model': 'ct', 'length': 40, 'turnrate': 360/200},
                {'model': 'cv', 'length': 30, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 90/20},
                {'model': 'cv', 'length': 30, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 90/20},
                {'model': 'cv', 'length': 40, 'velocity': 300},
                {'model': 'ct', 'length': 4, 'turnrate': 10/20},
                {'model': 'cv', 'length': 20, 'velocity': 300},
            ],
        ],
        'noise': [model.R_cv(3, [0., 0., 0.])] * 5,
        'pd': [1] * 5,
        'entries': 5
    }
    trajs_state, trajs_meas = model.trajectory_generator(record, seed=int(time.time()))
    trajs_state = trajs_state[2]        # center

    for i in range(5):
        origin = trajs_meas[2][10 + i, :-1]
        A = utils.rotate_matrix_deg(-18 * (i + 1))
        for j in range(5):
            trajs_meas[j][10 + i, :-1] = origin + (A @ (trajs_meas[j][10 + i, :-1] - origin))
    for i in range(75):
        origin = trajs_meas[2][15 + i, :-1]
        A = utils.rotate_matrix_deg(90)
        for j in range(5):
            trajs_meas[j][15 + i, :-1] = origin + (A @ (trajs_meas[j][15 + i, :-1] - origin))
    for i in range(5):
        origin = trajs_meas[2][90 + i, :-1]
        A = utils.rotate_matrix_deg(-90 + 18 * (i + 1))
        for j in range(5):
            trajs_meas[j][90 + i, :-1] = origin + (A @ (trajs_meas[j][90 + i, :-1] - origin))

    R = model.R_cv(3, [500, 100, 0])
    pd = 0.9
    N = trajs_state.shape[0]
    # print(N)

    # add noise
    for i in range(5):
        noi = st.multivariate_normal.rvs(cov=R, size=N)
        trajs_meas[i] += noi

    # remove some measurements according to `pd`
    for i in range(5):
        remove = st.uniform.rvs(size=N - 1) >= pd
        trajs_meas[i][1:][remove] = np.nan

    return trajs_state, trajs_meas

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
    trajs_state, trajs_meas = [], []
    for i in range(500):
        state, meas = GroupFormation()
        trajs_state.append(state)
        trajs_meas.append(meas)
    io.savemat('gtt_data.mat', {'trajs_state': trajs_state, 'trajs_meas': trajs_meas})