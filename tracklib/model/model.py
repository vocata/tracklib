# -*- coding: utf-8 -*-
'''
REFERENCES:
[1] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation," New York: John Wiley and Sons, Inc, 2001.
[2] R. A. Singer, "Estimating Optimal Tracking Filter Performance for Manned Maneuvering Targets," in IEEE Transactions on Aerospace and Electronic Systems, vol. AES-6, no. 4, pp. 473-483, July 1970.
[3] X. Rong Li and V. P. Jilkov, "Survey of maneuvering target tracking. Part I. Dynamic models," in IEEE Transactions on Aerospace and Electronic Systems, vol. 39, no. 4, pp. 1333-1364, Oct. 2003.
[4] W. Koch, "Tracking and Sensor Data Fusion: Methodological Framework and Selected Applications," Heidelberg, Germany: Springer, 2014.
'''
from __future__ import division, absolute_import, print_function


__all__ = [
    'F_poly', 'F_singer', 'F_van_keuk', 'Q_poly_dc', 'Q_poly_dd', 'Q_singer',
    'Q_van_keuk', 'H_pos_only', 'R_pos_only', 'F_cv', 'f_cv', 'f_cv_jac',
    'Q_cv_dc', 'Q_cv_dd', 'H_cv', 'h_cv', 'h_cv_jac', 'R_cv', 'F_ca', 'f_ca',
    'f_ca_jac', 'Q_ca_dc', 'Q_ca_dd', 'H_ca', 'h_ca', 'h_ca_jac', 'R_ca',
    'F_ct', 'f_ct', 'f_ct_jac', 'Q_ct', 'h_ct', 'h_ct_jac', 'R_ct',
    'model_switch', 'Trajectory'
]

import numbers
import numpy as np
import scipy.linalg as lg
from collections.abc import Iterable
from scipy.special import factorial
from tracklib.utils import multi_normal, cart2sph


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
        Singer's model is between 5 and 20 seconds. Typical values of tau for aircraft
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
    aT = alpha * T
    eaT = np.exp(-aT)
    F[0, 0] = 1
    F[0, 1] = T
    F[0, 2] = (aT - 1 + eaT) * tau**2
    F[1, 1] = 1
    F[1, 2] = (1 - eaT) * tau
    F[2, 2] = eaT
    F = np.kron(np.eye(axis), F_base)

    return F


def F_van_keuk(axis, T, tau=20):
    '''
    Get the state transition matrix for the van Keuk dynamic model. This is a
    direct discrete-time model such that the acceleration advances in each
    dimension over time as a[k+1]=exp(-T/tau)a[k]+std*sqrt(1-exp(-2*T/tau))*v[k],
    see section 2.2.1 in [4]
    
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
        Singer's model is between 5 and 20 seconds. Typical values of tau for aircraft
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

    F_base = F_poly(3, 1, T)
    F_base[-1, -1] = np.exp(-T / tau)
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

    if isinstance(std, numbers.Number):
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
        ht means that the order of the noise is `ht` greater than the highest order
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

    if isinstance(std, numbers.Number):
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
        Singer's model is between 5 and 20 seconds. Typical values of tau for aircraft
        are 20s for slow turn and 5s for an evasive maneuver. If this parameter
        is omitted, the default value of 20 is used.The time constant is assumed
        the same for all dimensions of motion, so this parameter is scalar.
    std : number, list
        std is the instantaneous standard deviation of the acceleration knowm as
        Ornstein-Uhlenbeck process, which can be obtained by assuming it to be
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

    if isinstance(std, numbers.Number):
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


def Q_van_keuk(axis, T, tau, std):
    '''
    Process noise covariance matrix for a Van Keuk dynamic model, see section 2.2.1 in [4]

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
        Singer's model is between 5 and 20 seconds. Typical values of tau for aircraft
        are 20s for slow turn and 5s for an evasive maneuver. If this parameter
        is omitted, the default value of 20 is used.The time constant is assumed
        the same for all dimensions of motion, so this parameter is scalar.
    std : number, list
        std is the instantaneous standard deviation of the acceleration knowm as
        Ornstein-Uhlenbeck process, which can be obtained by assuming it to be
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

    if isinstance(std, numbers.Number):
        std = [std] * axis
    Q_base = np.diag([0., 0., 1.])
    Q_base = (1 - np.exp(-2 * T / tau)) * Q_base
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

    if isinstance(std, numbers.Number):
        std = [std] * axis
    R = np.diag(std)**2

    return R


def F_cv(axis, T):
    return F_poly(2, axis, T)


def f_cv(axis, T):
    F = F_cv(axis, T)
    def f(x, u):
        return np.dot(F, x)
    return f


def f_cv_jac(axis, T):
    F = F_cv(axis, T)
    def fjac(x, u):
        return F
    return fjac


def Q_cv_dc(axis, T, std):
    return Q_poly_dc(2, axis, T, std)


def Q_cv_dd(axis, T, std):
    return Q_poly_dd(2, axis, T, std, ht=1)


def H_cv(axis):
    return H_pos_only(2, axis)


def h_cv(axis):
    H = H_cv(axis)
    def h(x):
        return np.dot(H, x)
    return h


def h_cv_jac(axis):
    H = H_cv(axis)
    def hjac(x):
        return H
    return hjac


def R_cv(axis, std):
    return R_pos_only(axis, std)


def F_ca(axis, T):
    return F_poly(3, axis, T)


def f_ca(axis, T):
    F = F_ca(axis, T)
    def f(x, u):
        return np.dot(F, x)
    return f


def f_ca_jac(axis, T):
    F = F_ca(axis, T)
    def fjac(x, u):
        return F
    return fjac


def Q_ca_dc(axis, T, std):
    return Q_poly_dc(3, axis, T, std)


def Q_ca_dd(axis, T, std):
    return Q_poly_dd(3, axis, T, std, ht=0)


def H_ca(axis):
    return H_pos_only(3, axis)


def h_ca(axis):
    H = H_ca(axis)
    def h(x):
        return np.dot(H, x)
    return h


def h_ca_jac(axis):
    H = H_ca(axis)
    def hjac(x):
        return H
    return hjac


def R_ca(axis, std):
    return R_pos_only(axis, std)


def F_ct(axis, turn_rate, T):
    assert (axis >= 2)

    omega = np.deg2rad(turn_rate)
    wt = omega * T
    sin_wt = np.sin(wt)
    cos_wt = np.cos(wt)
    if np.fabs(omega) >= np.sqrt(np.finfo(omega).eps):
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


def f_ct(axis, T):
    assert (axis >= 2)

    def f(x, u):
        omega = np.deg2rad(x[4])
        wt = omega * T
        sin_wt = np.sin(wt)
        cos_wt = np.cos(wt)
        if np.fabs(omega) >= np.sqrt(np.finfo(omega).eps):
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


def f_ct_jac(axis, T):
    assert (axis >= 2)

    def fjac(x, u):
        omega = np.deg2rad(x[4])
        wt = omega * T
        sin_wt = np.sin(wt)
        cos_wt = np.cos(wt)
        if np.fabs(omega) >= np.sqrt(np.finfo(omega).eps):
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


def Q_ct(axis, T, std):
    assert (axis >= 2)

    if isinstance(std, numbers.Number):
        std = [std] * (axis + 1)    # omega
    block = np.array([T**2 / 2, T], dtype=float).reshape(-1, 1)
    L = lg.block_diag(block, block, T)
    Q = np.diag(std)**2
    if axis == 3:
        L = lg.block_diag(L, block)
    return L @ Q @ L.T


def h_ct(axis):
    assert (axis >= 2)

    def h(x):
        if axis == 3:
            H = H_pos_only(2, 3)
        else:
            H = H_pos_only(2, 2)
        H = np.insert(H, 4, 0, axis=1)
        return np.dot(H, x)
    return h


def h_ct_jac(axis):
    assert (axis >= 2)

    def hjac(x):
        if axis == 3:
            H = H_pos_only(2, 3)
        else:
            H = H_pos_only(2, 2)
        H = np.insert(H, 4, 0, axis=1)
        return H
    return hjac


def R_ct(axis, std):
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
            sel = np.setdiff1d(range(ca_dim), range(2, ca_dim, 3))
            slct = np.eye(ca_dim)[:, sel]
            stmp = np.dot(slct, state)
            return stmp
        elif type_out == 'ct':
            slct = np.eye(5, 4)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            stmp = np.dot(slct, state)
            return stmp
        else:
            raise ValueError('unknown output type: %s' % type_out)
    elif type_in == 'ca':
        axis = dim // 3
        if type_out == 'cv':
            ca_dim = 3 * axis
            sel = np.setdiff1d(range(ca_dim), range(2, ca_dim, 3))
            slct = np.eye(ca_dim)[sel]
            stmp = np.dot(slct, state)
            return stmp
        elif type_out == 'ca':
            return state
        elif type_out == 'ct':
            # ca to cv
            ca_dim = 3 * axis
            sel = np.setdiff1d(range(ca_dim), range(2, ca_dim, 3))
            slct = np.eye(ca_dim)[sel]
            stmp = np.dot(slct, state)
            # cv to ct
            slct = np.eye(5, 4)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            stmp = np.dot(slct, stmp)
            return stmp
        else:
            raise ValueError('unknown output type: %s' % type_out)
    elif type_in == 'ct':
        axis = dim // 2
        if type_out == 'cv':
            slct = np.eye(4, 5)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            stmp = np.dot(slct, state)
            return stmp
        elif type_out == 'ca':
            # ct to cv
            slct = np.eye(4, 5)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            stmp = np.dot(slct, state)
            # cv to ca
            ca_dim = 3 * axis
            sel = np.setdiff1d(range(ca_dim), range(2, ca_dim, 3))
            slct = np.eye(ca_dim)[:, sel]
            stmp = np.dot(slct, stmp)
            return stmp
        elif type_out == 'ct':
            return state
        else:
            raise ValueError('unknown output type: %s' % type_out)
    else:
        raise ValueError('unknown input type: %s' % type_in)


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
            sel_diff = range(2, ca_dim, 3)
            sel = np.setdiff1d(range(ca_dim), sel_diff)
            slct = np.eye(ca_dim)[:, sel]
            ctmp = slct @ cov @ slct.T
            ctmp[sel_diff, sel_diff] = uncertainty
            return ctmp
        elif type_out == 'ct':
            slct = np.eye(5, 4)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            ctmp = slct @ cov @ slct.T
            ctmp[4, 4] = uncertainty
            return ctmp
        else:
            raise ValueError('unknown output type: %s' % type_out)
    elif type_in == 'ca':
        axis = dim // 3
        if type_out == 'cv':
            ca_dim = 3 * axis
            sel = np.setdiff1d(range(ca_dim), range(2, ca_dim, 3))
            slct = np.eye(ca_dim)[sel]
            ctmp = slct @ cov @ slct.T
            return ctmp
        elif type_out == 'ca':
            return cov
        elif type_out == 'ct':
            # ca to cv
            ca_dim = 3 * axis
            sel = np.setdiff1d(range(ca_dim), range(2, ca_dim, 3))
            slct = np.eye(ca_dim)[sel]
            ctmp = slct @ cov @ slct.T
            # cv to ct
            slct = np.eye(5, 4)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            ctmp = slct @ ctmp @ slct.T
            ctmp[4, 4] = uncertainty
            return ctmp
        else:
            raise ValueError('unknown output type: %s' % type_out)
    elif type_in == 'ct':
        axis = dim // 2
        if type_out == 'cv':
            slct = np.eye(4, 5)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            ctmp = slct @ cov @ slct.T
            return ctmp
        elif type_out == 'ca':
            # ct to cv
            slct = np.eye(4, 5)
            if axis == 3:
                slct = lg.block_diag(slct, np.eye(2))
            ctmp = slct @ cov @ slct.T
            # cv to ca
            ca_dim = 3 * axis
            sel_diff = range(2, ca_dim, 3)
            sel = np.setdiff1d(range(ca_dim), sel_diff)
            slct = np.eye(ca_dim)[:, sel]
            ctmp = slct @ ctmp @ slct.T
            ctmp[sel_diff, sel_diff] = uncertainty
            return ctmp
        elif type_out == 'ct':
            return cov
        else:
            raise ValueError('unknown output type: %s' % type_out)
    else:
        raise ValueError('unknown input type: %s' % type_in)


def model_switch(x, type_in, type_out):
    dim = len(x)
    if isinstance(x, np.ndarray):
        if len(x.shape) == 1:
            state = state_switch(x, type_in, type_out)
            return state
        elif len(x.shape) == 2:
            cov = cov_switch(x, type_in, type_out)
            return cov
        else:
            raise ValueError("shape of 'x' must be 1 or 2")
    elif hasattr(x, '__getitem__'):
        state = state_switch(x[0], type_in, type_out)
        cov = cov_switch(x[1], type_in, type_out)
        return state, cov
    else:
        raise TypeError("error 'x' type: '%s'" % x.__class__.__name__)


class Trajectory():
    def __init__(self, T, R, start=np.zeros(9), pd=None):
        assert (len(start) == 9)

        if R.shape == (3, 3):
            self._doppler = False
        elif R.shape == (4, 4):
            self._doppler = True
        else:
            raise ValueError('the shape of R must be (3, 3) or (4, 4)')
        if pd is None:
            self._pd = ()
        elif isinstance(pd, Iterable):
            self._pd = tuple(pd)
        else:
            raise TypeError("error 'pd' type: '%s'" % pd.__class__.__name__)

        self._T = T
        self._R = R
        self._head = start.copy()
        self._traj = []
        self._stage = []
        self._noise = None
        self._len = 0
        self._xdim = 9

    def __len__(self):
        return self._len

    def __call__(self, coordinate='xyz'):
        state = np.concatenate(self._traj, axis=1)

        H = H_ca(3)
        traj_real = np.dot(H, state)

        speed = np.empty(self._len)
        for i in range(self._len):
            p = state[0::3, i]
            v = state[1::3, i]
            d = lg.norm(p)
            speed[i] = np.dot(p, v) / d

        if coordinate == 'rae':
            meas = np.array(cart2sph(*state[::3]), dtype=float)
        elif coordinate == 'xyz':
            meas = traj_real
        else:
            raise ValueError("unknown coordinate: '%s'" % coordinate)

        if self._doppler:
            traj_real = np.vstack((traj_real, speed))
            meas = np.vstack((meas, speed))
        traj_meas = meas + self._noise

        for i in range(self._len):
            for scope, pd in self._pd:
                if scope.within(np.abs(speed[i])):
                    r = np.random.rand()
                    if r < 1 - pd:
                        traj_meas[:, i] = np.nan

        return traj_real, traj_meas, state

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
                if isinstance(v, numbers.Number):
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
                for j in range(traj_len):
                    if i == 0 and j == 0:
                        state[:, j] = self._head
                        continue
                    tmp = np.zeros(self._xdim)
                    tmp[sel] = np.dot(F, self._head[sel])
                    self._head[:] = tmp
                    state[:, j] = tmp
            elif mdl == 'ca':
                F = F_ca(3, self._T)
                a = stages[i]['acc']
                if isinstance(a, numbers.Number):
                    cur_v = self._head[[1, 4, 7]]
                    unit_v = cur_v / lg.norm(cur_v)
                    a *= unit_v
                if a[0] is not None:
                    self._head[2] = a[0]
                if a[1] is not None:
                    self._head[5] = a[1]
                if a[2] is not None:
                    self._head[8] = a[2]

                for j in range(traj_len):
                    if i == 0 and j == 0:
                        state[:, j] = self._head
                        continue
                    tmp = np.dot(F, self._head)
                    self._head[:] = tmp
                    state[:, j] = tmp
            elif mdl == 'ct':
                omega = stages[i]['omega']
                F = F_ct(3, omega, self._T)

                sel = [0, 1, 3, 4, 6, 7]
                for j in range(traj_len):
                    if i == 0 and j == 0:
                        state[:, j] = self._head
                        continue
                    tmp = np.zeros(self._xdim)
                    tmp[sel] = np.dot(F, self._head[sel])
                    self._head[:] = tmp
                    state[:, j] = tmp
            else:
                raise ValueError('unknown model')
            self._traj.append(state)

        self._noise = multi_normal(0, self._R, self._len, axis=1)
