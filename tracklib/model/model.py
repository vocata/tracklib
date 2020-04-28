# -*- coding: utf-8 -*-
'''
REFERENCES:
[1] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation," New York: John Wiley and Sons, Inc, 2001.
'''
from __future__ import division, absolute_import, print_function


__all__ = [
    'F_poly_trans', 'F_ct2D_trans', 'Q_dc_poly_proc_noise',
    'Q_dd_poly_proc_noise', 'Q_ct2D_proc_noise', 'H_only_pos_meas',
    'R_only_pos_meas_noise', 'Trajectory2D'
]

import numpy as np
import scipy.linalg as lg
from tracklib.utils import multi_normal
from scipy.special import factorial


def F_poly_trans(order, axis, T):
    '''
    This polynomial transition matrix is used with discretized continuous-time
    models as well as direct discrete-time models. see [1] section 6.2 and 6.3.

    Parameters
    ----------
    order : int
        The order >=0 of the filter. If order=1, then it is constant velocity,
        2 means constant acceleration, 3 means constant jerk, etc.
    axis : int
        Motion dimensions in Cartesian coordinate. If axis=0, it means x-axis,
        1 means x-axis and y-axis, etc.
    T : float
        The time-duration of the propagation interval.

    Returns
    -------
    F : ndarray
        The state transition matrix under a linear dynamic model of the given order
        and axis.
    '''
    assert (order >= 0)
    assert (axis >= 0)

    F_base = np.zeros((order + 1, order + 1))
    tmp = np.arange(order + 1)
    F_base[0, :] = T**tmp / factorial(tmp)
    for row in range(1, order + 1):
        F_base[row, row:] = F_base[0, :order - row + 1]
    F = np.kron(np.eye(axis + 1), F_base)

    return F


# F = F_poly_trans(2, 2, 3)
# print(F)


def F_ct2D_trans(turn_rate, T):
    w = turn_rate
    sin_val = np.sin(w * T)
    cos_val = np.cos(w * T)
    sin_rat = sin_val / w
    cos_rat = (cos_val - 1) / w
    F = np.array([[1, sin_rat, 0, cos_rat], [0, cos_val, 0, -sin_val],
                  [0, -cos_rat, 1, sin_rat], [0, sin_val, 0, cos_val]],
                 dtype=float)
    return F


# F = F_ct2D_trans(np.pi / 8, 1)
# print(F)


def Q_dc_poly_proc_noise(order, axis, T, std):
    '''
    Process noise covariance matrix used with discretized continuous-time models.
    see [1] section 6.2.

    Parameters
    ----------
    order : int
        The order >=0 of the filter. If order=1, then it is constant velocity,
        2 means constant acceleration, 3 means constant jerk, etc.
    axis : int
        Motion dimensions in Cartesian coordinate. If axis=0, it means x-axis,
        1 means x-axis and y-axis, etc.
    T : float
        The time-duration of the propagation interval.
    std : number, list or ndarray
        The standard deviation (square root of intensity) of continuous-time porcess noise

    Returns
    -------
    Q : ndarray
        Process noise convariance
    '''
    assert (order >= 0)
    assert (axis >= 0)

    if isinstance(std, (int, float)):
        std = [std] * (axis + 1)
    sel = np.arange(order, -1, -1)
    col, row = np.meshgrid(sel, sel)
    Q_base = T**(col + row + 1) / (factorial(col) * factorial(row) * (col + row + 1))
    Q = np.kron(np.diag(std)**2, Q_base)

    return Q


# Q = Q_dc_poly_proc_noise(2, 2, 3, [1, 2, 3])
# print(Q)


def Q_dd_poly_proc_noise(order, axis, T, std, ht=None):
    '''
    Process noise covariance matrix used with direct discrete-time models.
    see [1] section 6.3.

    Parameters
    ----------
    order : int
        The order >=0 of the filter. If order=1, then it is constant velocity,
        2 means constant acceleration, 3 means constant jerk, etc.
    axis : int
        Motion dimensions in Cartesian coordinate. If axis=0, it means x-axis,
        1 means x-axis and y-axis, etc.
    T : float
        The time-duration of the propagation interval.
    std : number, list or ndarray
        The standard deviation of discrete-time porcess noise
    ht : int
        ht means that the order of the noise is 'ht' greater than the highest order
        of the state, e.g., if the highest order of state is acceleration, then ht=0
        means that the noise is of the same order as the highest order of state, that
        is, the noise is acceleration and the model is DWPA, see[1] section 6.3.3. If
        the highest order is velocity, the ht=1 means the noise is acceleration and
        the model is DWNA, see[1] section 6.3.2.

    Returns
    -------
    Q : ndarray
        Process noise convariance

    Notes
    -----
    For the model to which the alpha filter applies, we have order=0, ht=2.
    Likewise, for the alpha-beta filter, order=1, ht=1 and for the alpha-
    beta-gamma filter, order=2, ht=0
    '''
    assert (order >= 0)
    assert (axis >= 0)

    if ht is None:
        ht = 0
    if isinstance(std, (int, float)):
        std = [std] * (axis + 1)

    sel = np.arange(ht + order, ht - 1, -1)
    L = T**sel / factorial(sel)
    Q_base = np.outer(L, L)
    Q = np.kron(np.diag(std)**2, Q_base)

    return Q


# Q = Q_dd_poly_proc_noise(1, 1, 1, [1, 2], 1)
# print(Q)


def Q_ct2D_proc_noise(T, std):
    '''
    Process noise covariance matrix used with coordinated turn model.
    see [1] section 11.7.

    Parameters
    ----------
    T : float
        The time-duration of the propagation interval.
    std : number, list or ndarray
        The standard deviation of discrete-time porcess noise

    Returns
    -------
    Q : ndarray
        Process noise convariance
    '''
    return Q_dd_poly_proc_noise(1, 1, T, std, 1)


# Q = Q_ct2D_proc_noise(0.38, [1, 2])
# print(Q)


def H_only_pos_meas(order, axis):
    '''
    Only-position measurement matrix is used with discretized continuous-time models
    as well as direct discrete-time models. see [1] section 6.5.

    Parameters
    ----------
    order : int
        The order >=0 of the filter. If order=1, then it is constant velocity,
        2 means constant acceleration, 3 means constant jerk, etc.
    axis : int
        Motion dimensions in Cartesian coordinate. If axis=0, it means x-axis,
        1 means x-axis and y-axis, etc.

    Returns
    -------
    H : ndarray
        the measurement or obervation matrix
    '''
    assert (order >= 0)
    assert (axis >= 0)

    H = np.eye((order + 1) * (axis + 1))
    H = H[::order + 1]

    return H


# H = H_only_pos_meas(1, 2)
# print(H)


def R_only_pos_meas_noise(axis, std):
    '''
    Only-position measurement noise covariance matrix and note that the noise
    of each axis are independent of each other. That is, R is diagonal matrix.
    '''
    if isinstance(std, (int, float)):
        std = [std] * (axis + 1)
    R = np.diag(std)**2

    return R


# R = R_only_pos_meas_noise(2, [1, 2, 3])
# print(R)


class Trajectory2D():
    def __init__(self, T, start=np.zeros(6)):
        self._T = T
        self._head = start.copy()
        self._state = []
        self._len = 0
        self._xdim = 6

    def __len__(self):
        return self._len

    def __call__(self, R):
        H = H_only_pos_meas(2, 1)
        state = np.concatenate(self._state, axis=1)
        traj_real = np.dot(H, state)
        v = multi_normal(0, R, self._len, axis=1)
        traj_meas = traj_real + v
        return traj_real, traj_meas

    def add_stage(self, stages):
        '''
        stage are list of dicts, for example:
        stage = [
            {'model': 'cv', 'len': 100, 'velocity': [30, 20]},
            {'model': 'ca', 'len': 100, 'acceleration': [10, 30]},
            {'model': 'ct', 'len': 100, 'omega': pi}
        ]
        '''
        for i in range(len(stages)):
            mdl = stages[i]['model']
            traj_len = stages[i]['len']
            self._len += traj_len
            state = np.zeros((self._xdim, traj_len))

            if mdl.lower() == 'cv':
                F = F_poly_trans(1, 1, self._T)
                v = stages[i]['velocity']
                if v[0] is not None:
                    self._head[1] = v[0]
                if v[1] is not None:
                    self._head[4] = v[1]

                sel = [0, 1, 3, 4]
                for i in range(traj_len):
                    self._head[sel] = np.dot(F, self._head[sel])
                    state[sel, i] = self._head[sel]
            elif mdl.lower() == 'ca':
                F = F_poly_trans(2, 1, self._T)
                a = stages[i]['acceleration']
                if a[0] is not None:
                    self._head[2] = a[0]
                if a[1] is not None:
                    self._head[5] = a[1]

                for i in range(traj_len):
                    self._head[:] = np.dot(F, self._head)
                    state[:, i] = self._head
            elif mdl.lower() == 'ct':
                F = F_ct2D_trans(stages[i]['omega'], self._T)

                sel = [0, 1, 3, 4]
                for i in range(traj_len):
                    self._head[sel] = np.dot(F, self._head[sel])
                    state[sel, i] = self._head[sel]
            else:
                raise ValueError('invalid model')

            self._state.append(state)

# import matplotlib.pyplot as plt
# start = np.array([100.0, 0.0, 0.0, 100.0, 0.0, 0.0], dtype=float)
# T = 0.1
# traj = Trajectory2D(T, start)
# stages = []
# stages.append({'model': 'cv', 'len': 100, 'velocity': [30, 0]})
# stages.append({'model': 'ct', 'len': 100, 'omega': np.deg2rad(300) / (100 * T)})
# stages.append({'model': 'cv', 'len': 100, 'velocity': [None, None]})
# stages.append({'model': 'ct', 'len': 100, 'omega': np.deg2rad(60) / (100 * T)})
# stages.append({'model': 'cv', 'len': 100, 'velocity': [None, None]})
# stages.append({'model': 'ct', 'len': 100, 'omega': np.deg2rad(90) / (100 * T)})
# stages.append({'model': 'ca', 'len': 100, 'acceleration': [0, 10]})
# traj.add_stage(stages)

# R = 10 * np.eye(2)
# traj_real, traj_meas = traj(R)

# # trajectory
# _, ax = plt.subplots()
# ax.scatter(traj_real[0, 0], traj_real[1, 0], s=120, c='r', marker='x', label='start')
# ax.plot(traj_real[0, :], traj_real[1, :], linewidth=0.8, label='real')
# ax.scatter(traj_meas[0, :], traj_meas[1, :], s=5, c='orange', label='meas')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.legend()
# ax.set_title('trajectory')
# plt.show()
