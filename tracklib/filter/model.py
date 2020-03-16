# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as lg

__all__ = ['newton_sys', 'SP_init', 'TPD_init']


def newton_sys(T, dim, axis):
    '''
    Return related matrices in Newtonian dynamic system

    Parameters
    ----------
    T : int or float 
        Sample interval
    dim : int
        Number of motion states in single axis
    axis : int
        Number of traget motion axes

    Returns
    -------
    F : ndarray
        Array of the state-transition matrix 
    L : ndarray
        Array of the process noise transition matrix
    H : ndarray
        Array of the measurement matrix 
    M : ndarray
        Array of the measurement noise transition matrix
    '''
    assert (0 < dim and dim <= 3)
    assert (0 < axis and axis <= 3)

    items = [1, T, T**2 / 2]

    F = np.zeros((0, dim * axis))
    L = np.zeros((0, axis))
    H = np.zeros((axis, 0))
    M = np.eye(H.shape[0])

    tmp = items.copy()
    for i in range(dim):
        F_rows = np.hstack(tuple(map(np.diag, map(lambda x: [x] * axis, tmp[:dim]))))
        F = np.vstack((F, F_rows))
        L_rows = np.diag([items[-1 - i]] * axis)
        L = np.vstack((L, L_rows))
        H_cols = np.eye(axis) if i == 0 else np.zeros((axis, axis))
        H = np.hstack((H, H_cols))

        tmp[-1] = 0
        # right cyclically shift one element
        tmp = tmp[-1:] + tmp[:-1]

    return F, L, H, M


# TODO
# 这只是在单个axis上的初始化
def SP_init(z, R, v_max):
    '''
    state and error convariance initiation using single-point method
    '''
    x = z
    x = np.vstack((x, np.zeros_like(x)))

    lt = R
    rb = v_max**2 / 3 * np.eye(*R.shape)
    P = lg.block_diag(lt, rb)

    return x, P


# TODO
# 这只是在单个axis上的初始化
def TPD_init(z1, z2, R1, R2, T):
    '''
    state and error convariance initiation using two-point
    difference method. The initial state error covariance is
    computed assuming there is no process noise.
    '''
    x = np.vstack((z2, z1))

    lt = R2
    lb = R2 / T
    rt = R2 / T
    rb = (R1 + R2) / T**2
    P = np.hstack((np.vstack((lt, lb)), np.vstack((rt, rb))))

    return x, P
