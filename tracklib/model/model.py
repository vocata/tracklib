# -*- coding: utf-8 -*-
'''
REFERENCES:
[1] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation," New York: John Wiley and Sons, Inc, 2001.
[2] R. A. Singer, "Estimating Optimal Tracking Filter Performance for Manned Maneuvering Targets," in IEEE Transactions on Aerospace and Electronic Systems, vol. AES-6, no. 4, pp. 473-483, July 1970.
[3] X. Rong Li and V. P. Jilkov, "Survey of maneuvering target tracking. Part I. Dynamic models," in IEEE Transactions on Aerospace and Electronic Systems, vol. 39, no. 4, pp. 1333-1364, Oct. 2003.
[4] W. Koch, "Tracking and Sensor Data Fusion: Methodological Framework and Selected Applications," Heidelberg, Germany: Springer, 2014.
[5] Mo Longbin, Song Xiaoquan, Zhou Yiyu, Sun Zhong Kang and Y. Bar-Shalom, "Unbiased converted measurements for tracking," in IEEE Transactions on Aerospace and Electronic Systems, vol. 34, no. 3, pp. 1023-1027, July 1998
'''
from __future__ import division, absolute_import, print_function


__all__ = [
    'F_poly', 'F_singer', 'F_van_keuk', 'Q_poly_dc', 'Q_poly_dd', 'Q_singer',
    'Q_van_keuk', 'H_pos_only', 'R_pos_only', 'F_cv', 'f_cv', 'f_cv_jac',
    'Q_cv_dc', 'Q_cv_dd', 'H_cv', 'h_cv', 'h_cv_jac', 'R_cv', 'F_ca', 'f_ca',
    'f_ca_jac', 'Q_ca_dc', 'Q_ca_dd', 'H_ca', 'h_ca', 'h_ca_jac', 'R_ca',
    'F_ct', 'f_ct', 'f_ct_jac', 'Q_ct', 'h_ct', 'h_ct_jac', 'R_ct',
    'convert_meas', 'model_switch', 'trajectory_cv', 'trajectory_ca',
    'trajectory_ct', 'trajectory_generator', 'trajectory_with_pd',
    'trajectory_to_meas'
]

import numbers
import numpy as np
import scipy.linalg as lg
import scipy.stats as st
import scipy.special as sl
from tracklib.utils import sph2cart, pol2cart


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
    F_base[0, :] = T**tmp / sl.factorial(tmp)
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
    F_base[0, 0] = 1
    F_base[0, 1] = T
    F_base[0, 2] = (aT - 1 + eaT) * tau**2
    F_base[1, 1] = 1
    F_base[1, 2] = (1 - eaT) * tau
    F_base[2, 2] = eaT
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
    Q_base = T**(col + row + 1) / (sl.factorial(col) * sl.factorial(row) * (col + row + 1))
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
    L = T**sel / sl.factorial(sel)
    Q_base = np.outer(L, L)
    Q = np.kron(np.diag(std)**2, Q_base)

    return Q


def Q_singer(axis, T, std, tau=20):
    '''
    Process noise covariance matrix used with Singer models. see section 8.2 in [1]

    Parameters
    ----------
    axis : int
        Motion directions in Cartesian coordinate. If axis=1, it means x-axis,
        2 means x-axis and y-axis, etc.
    T : float
        The time-duration of the propagation interval.
    std : number, list
        std is the instantaneous standard deviation of the acceleration knowm as
        Ornstein-Uhlenbeck process, which can be obtained by assuming it to be
        1. Equal to a maxmum acceleration a_M with probability p_M and -a_M with the same
           probability
        2. Equal to zero with probability p_0
        3. Uniformly distributed in [-a_M, a_M] with the remaining probability mass
        All parameters mentioned above are chosen by the designer. So the expected std^2
        is (a_M^2 / 3)*(1 + 4*p_M - p_0)
    tau : float
        The time constant of the target acceleration autocorrelation, that is, the
        decorrelation time is approximately 2*tau. A reasonable range of tau for
        Singer's model is between 5 and 20 seconds. Typical values of tau for aircraft
        are 20s for slow turn and 5s for an evasive maneuver. If this parameter
        is omitted, the default value of 20 is used.The time constant is assumed
        the same for all dimensions of motion, so this parameter is scalar.

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


def Q_van_keuk(axis, T, std, tau=20):
    '''
    Process noise covariance matrix for a Van Keuk dynamic model, see section 2.2.1 in [4]

    Parameters
    ----------
    axis : int
        Motion directions in Cartesian coordinate. If axis=1, it means x-axis,
        2 means x-axis and y-axis, etc.
    T : float
        The time-duration of the propagation interval.
    std : number, list
        std is the instantaneous standard deviation of the acceleration knowm as
        Ornstein-Uhlenbeck process, which can be obtained by assuming it to be
        1. Equal to a maxmum acceleration a_M with probability p_M and -a_M with the same
           probability
        2. Equal to zero with probability p_0
        3. Uniformly distributed in [-a_M, a_M] with the remaining probability mass
        All parameters mentioned above are chosen by the designer. So the expected std^2
        is (a_M^2 / 3)*(1 + 4*p_M - p_0)
    tau : float
        The time constant of the target acceleration autocorrelation, that is, the
        decorrelation time is approximately 2*tau. A reasonable range of tau for
        Singer's model is between 5 and 20 seconds. Typical values of tau for aircraft
        are 20s for slow turn and 5s for an evasive maneuver. If this parameter
        is omitted, the default value of 20 is used. The time constant is assumed
        the same for all dimensions of motion, so this parameter is scalar.

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
    def f(x, u=None):
        return np.dot(F, x)
    return f


def f_cv_jac(axis, T):
    F = F_cv(axis, T)
    def fjac(x, u=None):
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
    def f(x, u=None):
        return np.dot(F, x)
    return f


def f_ca_jac(axis, T):
    F = F_ca(axis, T)
    def fjac(x, u=None):
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


def F_ct(axis, turnrate, T):
    assert (axis >= 2)

    omega = np.deg2rad(turnrate)
    if np.fabs(omega) >= np.sqrt(np.finfo(omega).eps):
        wt = omega * T
        sin_wt = np.sin(wt)
        cos_wt = np.cos(wt)
        sin_div = sin_wt / omega
        cos_div = (cos_wt - 1) / omega
    else:
        sin_wt = 0
        cos_wt = 1
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

    def f(x, u=None):
        omega = np.deg2rad(x[4])
        if np.fabs(omega) >= np.sqrt(np.finfo(omega).eps):
            wt = omega * T
            sin_wt = np.sin(wt)
            cos_wt = np.cos(wt)
            sin_div = sin_wt / omega
            cos_div = (cos_wt - 1) / omega
        else:
            sin_wt = 0
            cos_wt = 1
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

    def fjac(x, u=None):
        omega = np.deg2rad(x[4])
        if np.fabs(omega) >= np.sqrt(np.finfo(omega).eps):
            wt = omega * T
            sin_wt = np.sin(wt)
            cos_wt = np.cos(wt)
            sin_div = sin_wt / omega
            cos_div = (cos_wt - 1) / omega
            f0 = np.deg2rad(((wt * cos_wt - sin_wt) * x[1] + (1 - cos_wt - wt * sin_wt) * x[3]) / omega**2)
            f1 = np.deg2rad((-x[1] * sin_wt - x[3] * cos_wt) * T)
            f2 = np.deg2rad((wt * (x[1] * sin_wt + x[3] * cos_wt) - (x[1] * (1 - cos_wt) + x[3] * sin_wt)) / omega**2)
            f3 = np.deg2rad((x[1]*cos_wt - x[3]*sin_wt) * T)
        else:
            sin_wt = 0
            cos_wt = 1
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

    if axis == 3:
        H = H_pos_only(2, 3)
    else:
        H = H_pos_only(2, 2)
    H = np.insert(H, 4, 0, axis=1)
    def h(x):
        return np.dot(H, x)
    return h


def h_ct_jac(axis):
    assert (axis >= 2)

    if axis == 3:
        H = H_pos_only(2, 3)
    else:
        H = H_pos_only(2, 2)
    H = np.insert(H, 4, 0, axis=1)
    def hjac(x):
        return H
    return hjac


def R_ct(axis, std):
    assert (axis >= 2)

    return R_pos_only(axis, std)


def convert_meas(z, R, elev=False):
    if elev:
        # coverted measurement
        r, az, el = z[0], z[1], z[2]
        var_r, var_az, var_el = R[0, 0], R[1, 1], R[2, 2]
        lamb_az = np.exp(-var_az / 2)
        lamb_el = np.exp(-var_el / 2)
        z_cart = np.array(sph2cart(r, az, el), dtype=float)
        z_cart[0] = z_cart[0] / lamb_az / lamb_el
        z_cart[1] = z_cart[1] / lamb_az / lamb_el
        z_cart[2] = z_cart[2] / lamb_el
        # coverted covariance
        r11 = (1 / (lamb_az * lamb_el)**2 - 2) * (r * np.cos(az) * np.cos(el))**2 + (r**2 + var_r) * (1 + lamb_az**4 * np.cos(2 * az)) * (1 + lamb_el**4 * np.cos(2 * el)) / 4
        r22 = (1 / (lamb_az * lamb_el)**2 - 2) * (r * np.sin(az) * np.cos(el))**2 + (r**2 + var_r) * (1 - lamb_az**4 * np.cos(2 * az)) * (1 + lamb_el**4 * np.cos(2 * el)) / 4
        r33 = (1 / lamb_el**2 - 2) * (r * np.sin(el))**2 + (r**2 + var_r) * (1 - lamb_el**4 * np.cos(2 * el)) / 2
        r12 = (1 / (lamb_az * lamb_el)**2 - 2) * r**2 * np.sin(az) * np.cos(az) * np.cos(el)**2 + (r**2 + var_r) * lamb_az**4 * np.sin(2 * az) * (1 + lamb_el**4 * np.cos(2 * el)) / 4
        r13 = (1 / (lamb_az * lamb_el**2) - 1 / lamb_az - lamb_az) * r**2 * np.cos(az) * np.sin(el) * np.cos(el) + (r**2 + var_r) * lamb_az * lamb_el**4 * np.cos(az) * np.sin(2 * el) / 2
        r23 = (1 / (lamb_az * lamb_el**2) - 1 / lamb_az - lamb_az) * r**2 * np.sin(az) * np.sin(el) * np.cos(el) + (r**2 + var_r) * lamb_az * lamb_el**4 * np.sin(az) * np.sin(2 * el) / 2
        R_cart = np.array([[r11, r12, r13], [r12, r22, r23], [r13, r23, r33]], dtype=float)
    else:
        # coverted measurement
        r, az = z[0], z[1]
        var_r, var_az = R[0, 0], R[1, 1]
        lamb_az = np.exp(-var_az / 2)
        z_cart = np.array(pol2cart(r, az), dtype=float) / lamb_az
        # coverted covariance
        r11 = (r**2 + var_r) / 2 * (1 + lamb_az**4 * np.cos(2 * az)) + (1 / lamb_az**2 - 2) * (r * np.cos(az))**2
        r22 = (r**2 + var_r) / 2 * (1 - lamb_az**4 * np.cos(2 * az)) + (1 / lamb_az**2 - 2) * (r * np.sin(az))**2
        r12 = (r**2 + var_r) / 2 * lamb_az**4 * np.sin(2 * az) + (1 / lamb_az**2 - 2) * r**2 * np.sin(az) * np.cos(az)
        R_cart = np.array([[r11, r12], [r12, r22]], dtype=float)
    return z_cart, R_cart


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


def trajectory_cv(state, interval, length, velocity):
    head = state.copy()
    dim = head.size
    order = 2
    axis = dim // order
    traj_cv = np.zeros((length, dim))

    vel = velocity
    cur_vel = head[1:dim:order]
    if isinstance(vel, numbers.Number):
        vel *= (cur_vel / lg.norm(cur_vel))
    else:
        vel = [cur_vel[i] if vel[i] is None else vel[i] for i in range(axis)]
    cur_vel[:] = vel        # it will also change the head
    head_cv = head

    F = F_cv(axis, interval)
    for i in range(length):
        head = np.dot(F, head)
        traj_cv[i] = head
    return traj_cv, head_cv

def trajectory_ca(state, interval, length, acceleration):
    head = state.copy()
    dim = state.size
    order = 3
    axis = dim // order
    traj_ca = np.zeros((length, dim))

    acc = acceleration
    cur_vel = head[1:dim:order]
    cur_acc = head[2:dim:order]
    if isinstance(acc, numbers.Number):
        acc *= (cur_vel / lg.norm(cur_vel))
    else:
        acc = [cur_acc[i] if acc[i] is None else acc[i] for i in range(axis)]
    cur_acc[:] = acc        # it will also change the head
    head_ca = head

    F = F_ca(axis, interval)
    for i in range(length):
        head = np.dot(F, head)
        traj_ca[i] = head
    return traj_ca, head_ca

def trajectory_ct(state, interval, length, turnrate, velocity=None):
    head = state.copy()
    dim = state.size
    order = 2
    axis = dim // order
    traj_ct = np.zeros((length, dim))

    if velocity is not None:
        vel = velocity
        cur_vel = head[1:dim:order]
        if isinstance(vel, numbers.Number):
            vel *= (cur_vel / lg.norm(cur_vel))
        else:
            vel = [cur_vel[i] if vel[i] is None else vel[i] for i in range(axis)]
        cur_vel[:] = vel
    head_ct = head

    F = F_ct(axis, turnrate, interval)
    for i in range(length):
        head = np.dot(F, head)
        traj_ct[i] = head
    return traj_ct, head_ct

def trajectory_generator(record):
    '''
    record = {
        'interval': [1, 1],
        'start':
        [
            [0, 0, 0],
            [0, 5, 0]
        ],
        'pattern':
        [
            [
                {'model': 'cv', 'length': 100, 'velocity': [250, 250, 0]},
                {'model': 'ct', 'length': 25, 'turnrate': 30}
            ],
            [
                {'model': 'cv', 'length': 100, 'velocity': [250, 250, 0]},
                {'model': 'ct', 'length': 30, 'turnrate': 30, 'velocity': 30}
            ]
        ],
        'noise':
        [
            10 * np.eye(3), 10 * np.eye(3)
        ],
        'pd':
        [
            0.9, 0.9
        ],
        'entries': 2
    }
    '''
    dim, order, axis = 9, 3, 3
    ca_sel = range(dim)
    acc_sel = range(2, dim, order)
    cv_sel = np.setdiff1d(ca_sel, acc_sel)
    ct_sel = np.setdiff1d(ca_sel, acc_sel)
    insert_sel = [2, 4, 6]

    interval = record['interval']
    start = record['start']
    pattern = record['pattern']
    noise = record['noise']
    entries = record['entries']

    trajs_state = []
    for i in range(entries):
        head = np.kron(start[i], [1., 0., 0.])
        state = np.kron(start[i], [1., 0., 0.]).reshape(1, -1)
        for pat in pattern[i]:
            if pat['model'] == 'cv':
                ret, head_cv = trajectory_cv(head[cv_sel], interval[i], pat['length'], pat['velocity'])
                ret = np.insert(ret, insert_sel, 0, axis=1)
                head = ret[-1, ca_sel]
                state[-1, acc_sel] = 0         # set the acceleration of previous state to zero
                state[-1, cv_sel] = head_cv    # change the velocity of previous state
                state = np.vstack((state, ret))
            elif pat['model'] == 'ca':
                ret, head_ca = trajectory_ca(head, interval[i], pat['length'], pat['acceleration'])
                head = ret[-1, ca_sel]
                state[-1, ca_sel] = head_ca    # change the acceleartion of previous state
                state = np.vstack((state, ret))
            elif pat['model'] == 'ct':
                if 'velocity' in pat:
                    ret, head_ct = trajectory_ct(head[ct_sel], interval[i], pat['length'], pat['turnrate'], pat['velocity'])
                else:
                    ret, head_ct = trajectory_ct(head[ct_sel], interval[i], pat['length'], pat['turnrate'])
                ret = np.insert(ret, insert_sel, 0, axis=1)
                head = ret[-1, ca_sel]
                state[-1, acc_sel] = 0
                state[-1, ct_sel] = head_ct
                state = np.vstack((state, ret))
            else:
                raise ValueError('invalid model')
        trajs_state.append(state)

    # add noise
    trajs_meas = []
    for i in range(entries):
        H = H_ca(axis)
        traj_len = trajs_state[i].shape[0]
        noi = st.multivariate_normal.rvs(cov=noise[i], size=traj_len)
        trajs_meas.append(np.dot(trajs_state[i], H.T) + noi)

    return trajs_state, trajs_meas


def trajectory_with_pd(trajs_meas, pd=0.8):
    for traj in trajs_meas:
        traj_len = traj.shape[0]
        remove_idx = st.uniform.rvs(size=traj_len) >= pd
        traj[remove_idx] = np.nan
    return trajs_meas


def trajectory_to_meas(trajs_meas, lamb=0):
    trajs_num = len(trajs_meas)
    min_x, max_x = np.inf, -np.inf
    min_y, max_y = np.inf, -np.inf
    min_z, max_z = np.inf, -np.inf
    max_traj_len = 0
    for traj in trajs_meas:
        min_x, max_x = min(min_x, traj[:, 0].min()), max(max_x, traj[:, 0].max())
        min_y, max_y = min(min_y, traj[:, 1].min()), max(max_y, traj[:, 1].max())
        min_z, max_z = min(min_z, traj[:, 2].min()), max(max_z, traj[:, 2].max())
        max_traj_len = max(max_traj_len, len(traj))
    trajs = []
    for i in range(max_traj_len):
        tmp = []
        for j in range(trajs_num):
            if i >= len(trajs_meas[j]) or np.any(np.isnan(trajs_meas[j][i])):
                continue
            tmp.append(trajs_meas[j][i])

        clutter_num = st.poisson.rvs(lamb)
        for j in range(clutter_num):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            z = np.random.uniform(min_z, max_z)
            tmp.append(np.array([x, y, z], dtype=float))
        tmp = np.array(tmp, dtype=float).reshape(-1, 3)
        trajs.append(tmp)
    return trajs
