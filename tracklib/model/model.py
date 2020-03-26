# -*- coding: utf-8 -*-
'''
REFERENCES:
[1] Mallick, M.,La Scala, B., "Comparison of single-point and two-point
    difference track initiation algorithms using position measurements". 
    Acta Automatica Sinica, 2008.
[2] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, Estimation with
    Applications to Tracking and Navigation. New York: John Wiley and
    Sons, Inc, 2001.
'''
from __future__ import division, absolute_import, print_function


__all__ = [
    'trans_mat', 'meas_mat', 'dc_proc_noise_cov', 'dd_proc_noise_cov',
    'meas_noise_cov', 'corr_noise'
]

import numpy as np
import scipy.linalg as lg
from scipy.special import factorial


def trans_mat(order, axis, T):
    '''
    This transition matrix is used with discretized continuous-time models
    as well as with direct discrete-time models.

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


# F = trans_mat(2, 3, 2)
# print(F)


def meas_mat(order, axis):
    '''
    This measurement matrix is used with discretized continuous-time models
    as well as with direct discrete-time models.

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


def dc_proc_noise_cov(order, axis, T, std):
    '''
    Construct a process noise covariance matrix used with discretized
    continuous-time models.

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


# Q = dc_proc_noise_cov(2, 2, 1, [1, 2, 3])
# print(Q)


def dd_proc_noise_cov(order, axis, T, std, ht=None):
    '''
    Construct a process noise covariance matrix used with direct discrete-time
    models.

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
        The order of the noise higher than the highest order in the state,
        e.g., if the highest order is acceleration, then ht=0 means that
        the noise is acceleration and the model is DWPA[2], and if the 
        highest order is velocity, the n=1 means the noise is acceleration
        and the model is DWNA[2]

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


# Q = dd_proc_noise_cov(2, 2, 1, [1, 2, 3], 1)
# print(Q)


def meas_noise_cov(axis, std):
    '''
    Construct a measurement noise covariance matrix used with discretized
    continuous-time models as well as with direct discrete-time models.
    '''
    if isinstance(std, (int, float)):
        std = [std] * (axis + 1)
    R = np.diag(std)**2

    return R


def corr_noise(cov, N=1):
    '''
    Generating zero-mean correlated Gaussian noise according to covariance matrix 'cov'

    Parameters
    ----------
    cov : ndarray
        Noise covariance matrix
    N : The number of noise

    Returns
    -------
    noi : ndarray
        Correlated Gaussian noise with mean zeros and covriance 'cov'.
    '''
    dim = cov.shape[0]
    e, v = lg.eigh(cov)
    std = np.sqrt(e)
    if N == 1:
        wgn = np.random.normal(size=(dim,))
        noi = std * wgn
    else:
        wgn = np.random.normal(size=(dim, N))
        noi = std.reshape(-1, 1) * wgn
    noi = np.dot(v, noi)
    return noi


# Q = np.diag([1, 2, 3])
# N = 1000
# Q_MC = 0
# mean_MC = 0
# for i in range(N):
#     noi = corr_noise(Q, 1000)
#     Q_MC += np.cov(noi)
#     mean_MC += np.mean(noi, axis=1)
# Q_MC = Q_MC / N
# mean_MC = mean_MC / N
# print(Q, Q_MC, sep='\n')
# print(mean_MC)