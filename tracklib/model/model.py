# -*- coding: utf-8 -*-
'''
REFERENCES:
[1] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation," New York: John Wiley and Sons, Inc, 2001.
'''
from __future__ import division, absolute_import, print_function


__all__ = [
    'F_poly_trans', 'Q_dc_ploy_proc_noise', 'Q_dd_poly_proc_noise',
    'H_only_pos_meas', 'R_only_pos_meas_noise'
]

import numpy as np
import scipy.linalg as lg
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
        2 means x-axis and y-axis, etc.
    T : float
        The time-duration of the propagation interval.

    Returns
    -------
    F : ndarray
        The state transition matrix under a linear dynamic model of the given order
        and axis.
    '''
    assert (0 <= order)
    assert (0 <= axis)

    F_1st = np.zeros((order + 1, order + 1))
    tmp = np.arange(order + 1)
    F_1st[0, :] = T**tmp / factorial(tmp)
    for row in range(1, order + 1):
        F_1st[row, row:] = F_1st[0, :order - row + 1]
    F = np.kron(F_1st, np.eye(axis + 1))

    return F


# F = F_poly_trans(2, 3, 2)
# print(F)


def Q_dc_ploy_proc_noise(order, axis, T, std):
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
        2 means x-axis and y-axis, etc.
    T : float
        The time-duration of the propagation interval.
    std : number, list or ndarray
        The standard deviation of continuous-time porcess noise

    Returns
    -------
    Q : ndarray
        Process noise convariance
    '''
    assert (0 <= order)
    assert (0 <= axis)

    if isinstance(std, (int, float)):
        std = [std] * (axis + 1)
    sel = np.arange(order, -1, -1)
    col, row = np.meshgrid(sel, sel)
    Q_base = T**(col + row + 1) / (factorial(col) * factorial(row) * (col + row + 1))
    Q = np.kron(Q_base, np.diag(std)**2)

    return Q


# Q = Q_dc_poly_proc_noise(2, 2, 1, [1, 2, 3])
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
        2 means x-axis and y-axis, etc.
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
    assert (0 <= order)
    assert (0 <= axis)

    if ht is None:
        ht = 0
    if isinstance(std, (int, float)):
        std = [std] * (axis + 1)

    sel = np.arange(ht + order, ht - 1, -1)
    L = T**sel / factorial(sel)
    Q_base = np.outer(L, L)
    Q = np.kron(Q_base, np.diag(std)**2)

    return Q


# Q = Q_dd_proc_noise(2, 2, 1, [1, 2, 3], 1)
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
        2 means x-axis and y-axis, etc.

    Returns
    -------
    H : ndarray
        the measurement or obervation matrix
    '''
    assert (0 <= order)
    assert (0 <= axis)

    H = np.zeros((axis + 1, (order + 1) * (axis + 1)))
    H[:axis + 1, :axis + 1] = np.eye(axis + 1)

    return H


def R_only_pos_meas_noise(axis, std):
    '''
    Only-position measurement noise covariance matrix and note that the noise
    of each axis are independent of each other. That is, R is diagonal matrix.
    '''
    if isinstance(std, (int, float)):
        std = [std] * (axis + 1)
    R = np.diag(std)**2

    return R
