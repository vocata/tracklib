# -*- coding: utf-8 -*-
'''
REFERENCES:
[1] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation," New York: John Wiley and Sons, Inc, 2001.
'''
from __future__ import division, absolute_import, print_function


__all__ = [
    'F_poly', 'F_singer', 'Q_poly_dc', 'Q_poly_dd', 'Q_singer', 'H_pos_only',
    'R_pos_only', 'F_cv', 'Q_cv_dc', 'Q_cv_dd', 'H_cv', 'R_cv', 'F_ca',
    'Q_ca_dc', 'Q_ca_dd', 'H_ca', 'R_ca', 'F_ct2D', 'f_ct2D', 'f_ct2D_jac',
    'Q_ct2D', 'h_ct2D', 'h_ct2D_jac', 'R_ct2D', 'model_switch', 'Trajectory2D'
]

import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tracklib.utils import multi_normal
from scipy.special import factorial


def F_poly(order, axis, T):
    '''
    This polynomial transition matrix is used with discretized continuous-time
    models as well as direct discrete-time models. see section 6.2 and 6.3 in [1].

    Parameters
    ----------
    order : int
        The order of the filter. If order=2, then it is constant velocity,
        3 means constant acceleration, 4 means constant jerk, etc.
    axis : int
        Motion directions in Cartesian coordinate. If axis=1, it means x-axis,
        2 means x-axis and y-axis, etc.
    T : float
        The time-duration of the propagation interval.

    Returns
    -------
    F : ndarray
        The state transition matrix under a linear dynamic model of the given order
        and axis.
    '''
    assert (order >= 1)
    assert (axis >= 1)

    F_base = np.zeros((order, order))
    tmp = np.arange(order)
    F_base[0, :] = T**tmp / factorial(tmp)
    for row in range(1, order):
        F_base[row, row:] = F_base[0, :order - row]
    F = np.kron(np.eye(axis), F_base)

    return F


def F_singer(axis, T, tau=20):
    '''
    Get the singer model transition matrix, see section 8.2 in [1].

    Parameters
    ----------
    axis : int
        Motion directions in Cartesian coordinate. If axis=1, it means x-axis,
        2 means x-axis and y-axis, etc.
    T : float
        The time-duration of the propagation interval.
    tau : float
        The time constant of the target acceleration autocorrelation, that is, the
        decorrelation time is approximately 2*tau. A reasonable range of tau for
        Singer's model is between 5 and 20 seconds. Typical values of tua for aircraft
        are 20s for slow turn and 5s for an evasive maneuver. If this parameter
        is omitted, the default value of 20 is used.The time constant is assumed
        the same for all dimensions of motion, so this parameter is scalar.
    Returns
    -------
    F : ndarray
        The state transition matrix under a Gauss-Markov dynamic model of the given
        axis.
    '''
    assert (axis >= 1)

    alpha = 1 / tau
    F_base = np.zeros((3, 3))
    F[0, 0] = 1
    F[0, 1] = T
    F[0, 2] = (alpha * T - 1 + np.exp(-alpha * T)) * tau**2
    F[1, 1] = 1
    F[1, 2] = (1 - np.exp(-alpha * T)) * tau
    F[2, 2] = np.exp(-alpha * T)
    F = np.kron(np.eye(axis), F_base)

    return F


def Q_poly_dc(order, axis, T, std):
    '''
    Process noise covariance matrix used with discretized continuous-time models.
    see section 6.2 in [1].

    Parameters
    ----------
    order : int
        The order of the filter. If order=2, then it is constant velocity,
        3 means constant acceleration, 4 means constant jerk, etc.
    axis : int
        Motion directions in Cartesian coordinate. If axis=1, it means x-axis,
        2 means x-axis and y-axis, etc.
    T : float
        The time-duration of the propagation interval.
    std : number, list
        The standard deviation (square root of intensity) of continuous-time porcess noise

    Returns
    -------
    Q : ndarray
        Process noise convariance
    '''
    assert (order >= 1)
    assert (axis >= 1)

    if isinstance(std, (int, float)):
        std = [std] * axis
    sel = np.arange(order - 1, -1, -1)
    col, row = np.meshgrid(sel, sel)
    Q_base = T**(col + row + 1) / (factorial(col) * factorial(row) * (col + row + 1))
    Q = np.kron(np.diag(std)**2, Q_base)

    return Q


def Q_poly_dd(order, axis, T, std, ht=0):
    '''
    Process noise covariance matrix used with direct discrete-time models.
    see section 6.3 in [1].

    Parameters
    ----------
    order : int
        The order of the filter. If order=2, then it is constant velocity,
        3 means constant acceleration, 4 means constant jerk, etc.
    axis : int
        Motion directions in Cartesian coordinate. If axis=1, it means x-axis,
        2 means x-axis and y-axis, etc.
    T : float
        The time-duration of the propagation interval.
    std : number, list
        The standard deviation of discrete-time porcess noise
    ht : int
        ht means that the order of the noise is 'ht' greater than the highest order
        of the state, e.g., if the highest order of state is acceleration, then ht=0
        means that the noise is of the same order as the highest order of state, that
        is, the noise is acceleration and the model is DWPA, see section 6.3.3 in [1].
        If the highest order is velocity, the ht=1 means the noise is acceleration and
        the model is DWNA, see section 6.3.2 in [1].

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
    assert (order >= 1)
    assert (axis >= 1)

    if isinstance(std, (int, float)):
        std = [std] * axis
    sel = np.arange(ht + order - 1, ht - 1, -1)
    L = T**sel / factorial(sel)
    Q_base = np.outer(L, L)
    Q = np.kron(np.diag(std)**2, Q_base)

    return Q


def Q_singer(axis, T, tau, std):
    '''
    Process noise covariance matrix used with Singer models. see section 8.2 in [1]

    Parameters
    ----------
    axis : int
        Motion directions in Cartesian coordinate. If axis=1, it means x-axis,
        2 means x-axis and y-axis, etc.
    T : float
        The time-duration of the propagation interval.
    tau : float
        The time constant of the target acceleration autocorrelation, that is, the
        decorrelation time is approximately 2*tau. A reasonable range of tau for
        Singer's model is between 5 and 20 seconds. Typical values of tua for aircraft
        are 20s for slow turn and 5s for an evasive maneuver. If this parameter
        is omitted, the default value of 20 is used.The time constant is assumed
        the same for all dimensions of motion, so this parameter is scalar.
    std : number, list
        std^2 is the instantaneous variance of the acceleration, which can be obtained
        by assuming it to be
        1. Equal to a maxmum acceleration a_M with probability p_M and -a_M with the same
           probability
        2. Equal to zero with probability p_0
        3. Uniformly distributed in [-a_M, a_M] with the remaining probability mass
        All parameters mentioned above are chosen by the designer. So the expected std^2
        is (a_M^2 / 3)*(1 + 4*p_M - p_0)

    Returns
    -------
    Q : ndarray
        Process noise convariance
    '''
    assert (axis >= 1)

    if isinstance(std, (int, float)):
        std = [std] * axis
    alpha = 1 / tau
    aT = alpha * T
    eaT = np.exp(-aT)
    e2aT = np.exp(-2 * aT)

    q11 = tau**4 * (1 - e2aT + 2 * aT + 2 * aT**3 / 3 - 2 * aT**2 - 4 * aT * eaT)
    q12 = tau**3 * (e2aT + 1 - 2 * eaT + 2 * aT * eaT - 2 * aT + aT**2)
    q13 = tau**2 * (1 - e2aT - 2 * aT * eaT)
    q22 = tau**2 * (4 * eaT - 3 - e2aT + 2 * aT)
    q23 = tau * (e2aT + 1 - 2 * eaT)
    q33 = 1 - e2aT
    Q_base = np.array([[q11, q12, q13],
                       [q12, q22, q23],
                       [q13, q23, q33]], dtype=float)
    Q = np.kron(np.diag(std)**2, Q_base)

    return Q


def H_pos_only(order, axis):
    '''
    Position-only measurement matrix is used with discretized continuous-time models
    as well as direct discrete-time models. see section 6.5 in [1].

    Parameters
    ----------
    order : int
        The order of the filter. If order=2, then it is constant velocity,
        3 means constant acceleration, 4 means constant jerk, etc.
    axis : int
        Motion directions in Cartesian coordinate. If axis=1, it means x-axis,
        2 means x-axis and y-axis, etc.

    Returns
    -------
    H : ndarray
        the measurement or obervation matrix
    '''
    assert (order >= 1)
    assert (axis >= 1)

    H = np.eye(order * axis)
    H = H[::order]

    return H


def R_pos_only(axis, std):
    '''
    Position-only measurement noise covariance matrix and the noise of each
    axis is assumed to be uncorrelated.

    Parameters
    ----------
    axis : int
        Motion directions in Cartesian coordinate. If axis=1, it means x-axis,
        2 means x-axis and y-axis, etc.

    Returns
    -------
    R : ndarray
        the measurement noise covariance matrix
    '''
    assert (axis >= 1)

    if isinstance(std, (int, float)):
        std = [std] * axis
    R = np.diag(std)**2

    return R


# specific model
def F_cv(axis, T):
    return F_poly(2, axis, T)


def Q_cv_dc(axis, T, std):
    return Q_poly_dc(2, axis, T, std)


def Q_cv_dd(axis, T, std):
    return Q_poly_dd(2, axis, T, std, ht=1)


def H_cv(axis):
    return H_pos_only(2, axis)


def R_cv(axis, std):
    return R_pos_only(axis, std)


def F_ca(axis, T):
    return F_poly(3, axis, T)


def Q_ca_dc(axis, T, std):
    return Q_poly_dc(3, axis, T, std)


def Q_ca_dd(axis, T, std):
    return Q_poly_dd(3, axis, T, std, ht=0)


def H_ca(axis):
    return H_pos_only(3, axis)


def R_ca(axis, std):
    return R_pos_only(axis, std)


def F_ct2D(axis, turn_rate, T):
    assert (axis >= 2)

    omega = np.deg2rad(turn_rate)
    wt = omega * T
    sin_wt = np.sin(wt)
    cos_wt = np.cos(wt)
    if omega != 0:
        sin_div = sin_wt / omega
        cos_div = (cos_wt - 1) / omega
    else:
        sin_div = T
        cos_div = 0
    F = np.array([[1, sin_div, 0, cos_div], [0, cos_wt, 0, -sin_wt],
                  [0, -cos_div, 1, sin_div], [0, sin_wt, 0, cos_wt]],
                 dtype=float)
    if axis == 3:
        zblock = F_cv(1, T)
        F = lg.block_diag(F, zblock)
    return F


def f_ct2D(axis, T):
    assert (axis >= 2)

    def f(x, u):
        omega = np.deg2rad(x[4])
        wt = omega * T
        sin_wt = np.sin(wt)
        cos_wt = np.cos(wt)
        if omega != 0:
            sin_div = sin_wt / omega
            cos_div = (cos_wt - 1) / omega
        else:
            sin_div = T
            cos_div = 0

        F = np.array([[1, sin_div, 0, cos_div], [0, cos_wt, 0, -sin_wt],
                      [0, -cos_div, 1, sin_div], [0, sin_wt, 0, cos_wt]],
                     dtype=float)
        F = lg.block_diag(F, 1)
        if axis == 3:
            zblock = F_cv(1, T)
            F = lg.block_diag(F, zblock)
        return np.dot(F, x)
    return f


def f_ct2D_jac(axis, T):
    assert (axis >= 2)

    def fjac(x, u):
        omega = np.deg2rad(x[4])
        wt = omega * T
        sin_wt = np.sin(wt)
        cos_wt = np.cos(wt)
        if omega != 0:
            sin_div = sin_wt / omega
            cos_div = (cos_wt - 1) / omega
            f0 = np.deg2rad(((wt * cos_wt - sin_wt) * x[1] + (1 - cos_wt - wt * sin_wt) * x[3]) / omega**2)
            f1 = np.deg2rad((-x[1] * sin_wt - x[3] * cos_wt) * T)
            f2 = np.deg2rad((wt * (x[1] * sin_wt + x[3] * cos_wt) - (x[1] * (1 - cos_wt) + x[3] * sin_wt)) / omega**2)
            f3 = np.deg2rad((x[1]*cos_wt - x[3]*sin_wt) * T)
        else:
            sin_div = T
            cos_div = 0
            f0 = np.deg2rad(-x[3] * T**2 / 2)
            f1 = np.deg2rad(-x[3] * T)
            f2 = np.deg2rad(x[1] * T**2 / 2)
            f3 = np.deg2rad(x[1] * T)

        F = np.array([[1, sin_div, 0, cos_div], [0, cos_wt, 0, -sin_wt],
                      [0, -cos_div, 1, sin_div], [0, sin_wt, 0, cos_wt]],
                     dtype=float)
        F = lg.block_diag(F, 1)
        F[0, -1] = f0
        F[1, -1] = f1
        F[2, -1] = f2
        F[3, -1] = f3
        if axis == 3:
            zblock = F_cv(1, T)
            F = lg.block_diag(F, zblock)
        return F
    return fjac


def Q_ct2D(axis, T, std):
    assert (axis >= 2)

    if isinstance(std, (int, float)):
        std = [std] * (axis + 1)    # omega
    block = np.array([T**2 / 2, T], dtype=float).reshape(-1, 1)
    L = lg.block_diag(block, block, T)
    Q = np.diag(std)**2
    if axis == 3:
        L = lg.block_diag(L, block)
    return L @ Q @ L.T


def h_ct2D(axis):
    assert (axis >= 2)

    def h(x):
        if axis == 3:
            H = H_pos_only(2, 3)
        else:
            H = H_pos_only(2, 2)
        H = np.insert(H, 4, 0, axis=1)
        return np.dot(H, x)
    return h


def h_ct2D_jac(axis):
    assert (axis >= 2)

    def hjac(x):
        if axis == 3:
            H = H_pos_only(2, 3)
        else:
            H = H_pos_only(2, 2)
        H = np.insert(H, 4, 0, axis=1)
        return H
    return hjac


def R_ct2D(axis, std):
    assert (axis >= 2)

    return R_pos_only(axis, std)


def state_switch(state, type_in, type_out):
    dim = len(state)
    state = state.copy()
    if type_in == 'cv':
        axis = dim // 2
        if type_out == 'cv':
            return state
        elif type_out == 'ca':
            ca_dim = 3 * axis
            sel = range(2, ca_dim, 3)
            slct = np.delete(np.eye(ca_dim), sel, axis=1)
            stmp = np.dot(slct, state)
            return stmp
        elif type_out == 'ct2D':
            slct = np.eye(5, 4)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            stmp = np.dot(slct, state)
            return stmp
        else:
            raise ValueError('unknown output type %s' % type_out)
    elif type_in == 'ca':
        axis = dim // 3
        if type_out == 'cv':
            ca_dim = 3 * axis
            sel = range(2, ca_dim, 3)
            slct = np.delete(np.eye(ca_dim), sel, axis=0)
            stmp = np.dot(slct, state)
            return stmp
        elif type_out == 'ca':
            return state
        elif type_out == 'ct2D':
            # ca to cv
            ca_dim = 3 * axis
            sel = range(2, ca_dim, 3)
            slct = np.delete(np.eye(ca_dim), sel, axis=0)
            stmp = np.dot(slct, state)
            # cv to ct2D
            slct = np.eye(5, 4)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            stmp = np.dot(slct, stmp)
            return stmp
        else:
            raise ValueError('unknown output type %s' % type_out)
    elif type_in == 'ct2D':
        axis = dim // 2
        if type_out == 'cv':
            slct = np.eye(4, 5)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            stmp = np.dot(slct, state)
            return stmp
        elif type_out == 'ca':
            # ct2D to cv
            slct = np.eye(4, 5)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            stmp = np.dot(slct, state)
            # cv to ca
            ca_dim = 3 * axis
            sel = range(2, ca_dim, 3)
            slct = np.delete(np.eye(ca_dim), sel, axis=1)
            stmp = np.dot(slct, stmp)
            return stmp
        elif type_out == 'ct2D':
            return state
        else:
            raise ValueError('unknown output type %s' % type_out)
    else:
        raise ValueError('unkonw input type %s' % type_in)


def cov_switch(cov, type_in, type_out):
    dim = len(cov)
    cov = cov.copy()
    uncertainty = 100
    if type_in == 'cv':
        axis = dim // 2
        if type_out == 'cv':
            return cov
        elif type_out == 'ca':
            ca_dim = 3 * axis
            sel = range(2, ca_dim, 3)
            slct = np.delete(np.eye(ca_dim), sel, axis=1)
            ctmp = slct @ cov @ slct.T
            ctmp[sel, sel] = uncertainty
            return ctmp
        elif type_out == 'ct2D':
            slct = np.eye(5, 4)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            ctmp = slct @ cov @ slct.T
            ctmp[4, 4] = uncertainty
            return ctmp
        else:
            raise ValueError('unknown output type %s' % type_out)
    elif type_in == 'ca':
        axis = dim // 3
        if type_out == 'cv':
            ca_dim = 3 * axis
            sel = range(2, ca_dim, 3)
            slct = np.delete(np.eye(ca_dim), sel, axis=0)
            ctmp = slct @ cov @ slct.T
            return ctmp
        elif type_out == 'ca':
            return cov
        elif type_out == 'ct2D':
            # ca to cv
            ca_dim = 3 * axis
            sel = range(2, ca_dim, 3)
            slct = np.delete(np.eye(ca_dim), sel, axis=0)
            ctmp = slct @ cov @ slct.T
            # cv to ct2D
            slct = np.eye(5, 4)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            ctmp = slct @ ctmp @ slct.T
            ctmp[4, 4] = uncertainty
            return ctmp
        else:
            raise ValueError('unknown output type %s' % type_out)
    elif type_in == 'ct2D':
        axis = dim // 2
        if type_out == 'cv':
            slct = np.eye(4, 5)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            ctmp = slct @ cov @ slct.T
            return ctmp
        elif type_out == 'ca':
            # ct2D to cv
            slct = np.eye(4, 5)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            ctmp = slct @ cov @ slct.T
            # cv to ca
            ca_dim = 3 * axis
            sel = range(2, ca_dim, 3)
            slct = np.delete(np.eye(ca_dim), sel, axis=1)
            ctmp = slct @ ctmp @ slct.T
            ctmp[sel, sel] = uncertainty
            return ctmp
        elif type_out == 'ct2D':
            return cov
        else:
            raise ValueError('unknown output type %s' % type_out)
    else:
        raise ValueError('unkonw input type %s' % type_in)


def model_switch(x, type_in, type_out):
    dim = len(x)
    if isinstance(x, (list, tuple)):
        state = state_switch(x[0], type_in, type_out)
        cov = cov_switch(x[1], type_in, type_out)
        return state, cov
    elif isinstance(x, np.ndarray):
        if len(x.shape) == 1:
            state = state_switch(x, type_in, type_out)
            return state
        elif len(x.shape) == 2:
            cov = cov_switch(x, type_in, type_out)
            return cov
        else:
            raise ValueError('x must be a 1-D array or 2-D array')
    else:
        raise ValueError('x must be list, tuple or ndarray')


class Trajectory2D():
    def __init__(self, T, start=np.zeros(9)):
        assert (len(start) == 9)

        self._T = T
        self._head = start.copy()
        self._traj = [start.copy().reshape(-1, 1)]
        self._stage = [{'model': 'start'}]
        self._len = 1
        self._xdim = 9

    def __len__(self):
        return self._len

    def __call__(self, R):
        H = H_ca(3)
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
            {'model': 'cv', 'len': 100, 'vel': [30, 20, 1]},
            {'model': 'cv', 'len': 100, 'vel': 7},
            {'model': 'ca', 'len': 100, 'acc': [10, 30, 1]},
            {'model': 'ca', 'len': 100, 'acc': 3},
            {'model': 'ct', 'len': 100, 'omega': 30}
        ]
        '''
        self._stage.extend(stages)
        for i in range(len(stages)):
            mdl = stages[i]['model']
            traj_len = stages[i]['len']
            self._len += traj_len

            state = np.zeros((self._xdim, traj_len))
            if mdl == 'cv':
                F = F_cv(3, self._T)
                v = stages[i]['vel']
                if isinstance(v, (int, float)):
                    cur_v = self._head[[1, 4, 7]]
                    unit_v = cur_v / lg.norm(cur_v)
                    v *= unit_v
                if v[0] is not None:
                    self._head[1] = v[0]
                if v[1] is not None:
                    self._head[4] = v[1]
                if v[2] is not None:
                    self._head[7] = v[2]

                sel = [0, 1, 3, 4, 6, 7]
                for i in range(traj_len):
                    self._head[sel] = np.dot(F, self._head[sel])
                    state[sel, i] = self._head[sel]
            elif mdl == 'ca':
                F = F_ca(3, self._T)
                a = stages[i]['acc']
                if isinstance(a, (int, float)):
                    cur_v = self._head[[1, 4, 7]]
                    unit_v = cur_v / lg.norm(cur_v)
                    a *= unit_v
                if a[0] is not None:
                    self._head[2] = a[0]
                if a[1] is not None:
                    self._head[5] = a[1]
                if a[2] is not None:
                    self._head[8] = a[2]

                for i in range(traj_len):
                    self._head[:] = np.dot(F, self._head)
                    state[:, i] = self._head
            elif mdl == 'ct':
                omega = stages[i]['omega']
                F = F_ct2D(3, omega, self._T)

                sel = [0, 1, 3, 4, 6, 7]
                for i in range(traj_len):
                    self._head[sel] = np.dot(F, self._head[sel])
                    state[sel, i] = self._head[sel]
            else:
                raise ValueError('invalid model')

            self._traj.append(state)

    def show_traj(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self._traj[0][0], self._traj[0][3], self._traj[0][6], s=50, c='r', marker='x', label=self._stage[0]['model'])
        for i in range(1, len(self._traj)):
            ax.plot(self._traj[i][0, :], self._traj[i][3, :], self._traj[i][6, :], '.-', linewidth=1, ms=3, label=self._stage[i]['model'])
        ax.legend()
        plt.show()
