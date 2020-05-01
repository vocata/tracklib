# -*- coding: utf-8 -*-
'''
REFERENCES:
[1] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation," New York: John Wiley and Sons, Inc, 2001.
'''
from __future__ import division, absolute_import, print_function


__all__ = [
    'F_poly', 'Q_poly_dc', 'Q_poly_dd', 'Q_ct2D', 'H_only_pos', 'F_cv',
    'Q_cv_dc', 'Q_cv_dd', 'H_cv', 'F_ca', 'Q_ca_dc', 'Q_ca_dd', 'H_ca',
    'F_ct2D', 'Q_ct2D', 'H_ct2D', 'state_switch', 'cov_switch', 'model_switch',
    'Trajectory2D'
]

import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
from tracklib.utils import multi_normal
from scipy.special import factorial


def F_poly(order, axis, T):
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


def Q_poly_dc(order, axis, T, std):
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


def Q_poly_dd(order, axis, T, std, ht=0):
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

    if isinstance(std, (int, float)):
        std = [std] * (axis + 1)

    sel = np.arange(ht + order, ht - 1, -1)
    L = T**sel / factorial(sel)
    Q_base = np.outer(L, L)
    Q = np.kron(np.diag(std)**2, Q_base)

    return Q


def H_only_pos(order, axis):
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


# def R_only_pos(axis, std):
#     '''
#     Only-position measurement noise covariance matrix and note that the noise
#     of each axis are independent of each other. That is, R is diagonal matrix.
#     '''
#     if isinstance(std, (int, float)):
#         std = [std] * (axis + 1)
#     R = np.diag(std)**2

#     return R


# specific model
def F_cv(axis, T):
    return F_poly(1, axis, T)


def Q_cv_dc(axis, T, std):
    return Q_poly_dc(1, axis, T, std)


def Q_cv_dd(axis, T, std):
    return Q_poly_dd(1, axis, T, std, ht=1)


def H_cv(axis):
    return H_only_pos(1, axis)


def F_ca(axis, T):
    return F_poly(2, axis, T)


def Q_ca_dc(axis, T, std):
    return Q_poly_dc(2, axis, T, std)


def Q_ca_dd(axis, T, std):
    return Q_poly_dd(2, axis, T, std, ht=0)


def H_ca(axis):
    return H_only_pos(2, axis)


def F_ct2D(turn_rate, T):
    w = turn_rate
    sin_val = np.sin(w * T)
    cos_val = np.cos(w * T)
    sin_rat = sin_val / w
    cos_rat = (cos_val - 1) / w
    F = np.array([[1, sin_rat, 0, cos_rat], [0, cos_val, 0, -sin_val],
                  [0, -cos_rat, 1, sin_rat], [0, sin_val, 0, cos_val]],
                 dtype=float)
    return F


def Q_ct2D(T, std):
    return Q_poly_dd(1, 1, T, std, ht=1)


def H_ct2D(axis):
    return H_only_pos(1, axis)



def state_switch(state, type_in, type_out):
    dim = len(state)
    if type_in == 'cv' or type_in == 'ct2D':
        axis = dim // 2
        if type_out == 'cv' or type_out == 'ct2D':
            return state.copy()
        elif type_out == 'ca':
            slct = np.delete(np.eye(3 * axis), np.s_[2::3], axis=1)
            stmp = np.dot(slct, state)
            return stmp
        else:
            raise ValueError('unknown output type %s' % type_out)
    elif type_in == 'ca':
        axis = dim // 3
        if type_out == 'cv' or type_out == 'ct2D':
            slct = np.delete(np.eye(3 * axis), np.s_[2::3], axis=0)
            stmp = np.dot(slct, state)
            return stmp
        elif type_out == 'ca':
            return state.copy()
        else:
            raise ValueError('unknown output type %s' % type_out)
    else:
        raise ValueError('unkonw input type %s' % type_in)


def cov_switch(cov, type_in, type_out):
    dim = len(cov)
    if type_in == 'cv' or type_in == 'ct2D':
        axis = dim // 2
        if type_out == 'cv' or type_out == 'ct2D':
            return cov.copy()
        elif type_out == 'ca':
            slct = np.delete(np.eye(3 * axis), np.s_[2::3], axis=1)
            ctmp = slct @ cov @ slct.T
            return ctmp
        else:
            raise ValueError('unknown output type %s' % type_out)
    elif type_in == 'ca':
        axis = dim // 3
        if type_out == 'cv' or type_out == 'ct2D':
            slct = np.delete(np.eye(3 * axis), np.s_[2::3], axis=0)
            ctmp = slct @ cov @ slct.T
            return ctmp
        elif type_out == 'ca':
            return cov.copy()
        else:
            raise ValueError('unknown output type %s' % type_out)
    else:
        raise ValueError('unkonw input type %s' % type_in)


def model_switch(state, cov, type_in, type_out):
    dim = len(state)
    if type_in == 'cv' or type_in == 'ct2D':
        axis = dim // 2
        if type_out == 'cv' or type_out == 'ct2D':
            return state.copy(), cov.copy()
        elif type_out == 'ca':
            slct = np.delete(np.eye(3 * axis), np.s_[2::3], axis=1)
            stmp = np.dot(slct, state)
            ctmp = slct @ cov @ slct.T
            return stmp, ctmp
        else:
            raise ValueError('unknown output type %s' % type_out)
    elif type_in == 'ca':
        axis = dim // 3
        if type_out == 'cv' or type_out == 'ct2D':
            slct = np.delete(np.eye(3 * axis), np.s_[2::3], axis=0)
            stmp = np.dot(slct, state)
            ctmp = slct @ cov @ slct.T
            return stmp, ctmp
        elif type_out == 'ca':
            return state.copy(), cov.copy()
        else:
            raise ValueError('unknown output type %s' % type_out)
    else:
        raise ValueError('unkonw input type %s' % type_in)


class Trajectory2D():
    def __init__(self, T, start=np.zeros(6)):
        self._T = T
        self._head = start.copy()
        self._traj = [start.copy().reshape(-1, 1)]
        self._stage = [{'model': 'start'}]
        self._len = 1
        self._xdim = 6

    def __len__(self):
        return self._len

    def __call__(self, R):
        H = H_only_pos(2, 1)
        state = np.concatenate(self._traj, axis=1)
        v = multi_normal(0, R, self._len, axis=1)
        traj_real = np.dot(H, state)
        traj_meas = traj_real + v
        return traj_real, traj_meas

    def stage(self):
        return self._stage

    def traj(self):
        return self._traj

    def add_stage(self, stages):
        '''
        stage are list of dicts, for example:
        stage = [
            {'model': 'cv', 'len': 100, 'velocity': [30, 20]},
            {'model': 'ca', 'len': 100, 'acceleration': [10, 30]},
            {'model': 'ct', 'len': 100, 'omega': pi}
        ]
        '''
        self._stage.extend(stages)
        for i in range(len(stages)):
            mdl = stages[i]['model']
            traj_len = stages[i]['len']
            self._len += traj_len

            state = np.zeros((self._xdim, traj_len))
            if mdl.lower() == 'cv':
                F = F_cv(1, self._T)
                v = stages[i]['velocity']
                if isinstance(v, (int, float)):
                    cur_v = self._head[[1, 4]]
                    unit_v = cur_v / lg.norm(cur_v)
                    v *= unit_v
                if v[0] is not None:
                    self._head[1] = v[0]
                if v[1] is not None:
                    self._head[4] = v[1]

                sel = [0, 1, 3, 4]
                for i in range(traj_len):
                    self._head[sel] = np.dot(F, self._head[sel])
                    state[sel, i] = self._head[sel]
            elif mdl.lower() == 'ca':
                F = F_ca(1, self._T)
                a = stages[i]['acceleration']
                if isinstance(a, (int, float)):
                    cur_v = self._head[[1, 4]]
                    unit_v = cur_v / lg.norm(cur_v)
                    a *= unit_v
                if a[0] is not None:
                    self._head[2] = a[0]
                if a[1] is not None:
                    self._head[5] = a[1]

                for i in range(traj_len):
                    self._head[:] = np.dot(F, self._head)
                    state[:, i] = self._head
            elif mdl.lower() == 'ct':
                F = F_ct2D(stages[i]['omega'], self._T)

                sel = [0, 1, 3, 4]
                for i in range(traj_len):
                    self._head[sel] = np.dot(F, self._head[sel])
                    state[sel, i] = self._head[sel]
            else:
                raise ValueError('invalid model')

            self._traj.append(state)

    def show_traj(self):
        _, ax = plt.subplots()
        ax.axis('equal')
        ax.scatter(self._traj[0][0], self._traj[0][3], s=50, c='r', marker='x', label=self._stage[0]['model'])
        for i in range(1, len(self._traj)):
            ax.plot(self._traj[i][0, :], self._traj[i][3, :], '.-', linewidth=1, ms=3, label=self._stage[i]['model'])
        ax.legend()
        plt.show()
