# -*- coding: utf-8 -*-
'''
REFERENCES:
[1]. Mallick, M.,La Scala, B., "Comparison of single-point and two-point difference track initiation algorithms using position measurements". Acta Automatica Sinica, 2008.
'''
from __future__ import division, absolute_import, print_function


__all__ = [
    'single_point_init', 'two_point_diff_init', 'biased_three_point_diff_init',
    'unbiased_three_point_diff_init'
]

import numpy as np


def __swap(state, cov, order):
    assert (order >= 0)

    order = order + 1
    stmp = np.zeros_like(state)
    for i in range(order):
        st = i * order
        ed = (i + 1) * order
        stmp[st:ed] = state[i::order]
    state = stmp

    ctmp_row = np.zeros_like(cov)
    for i in range(order):
        st = i * order
        ed = (i + 1) * order
        ctmp_row[st:ed] = cov[i::order]
    ctmp_col = ctmp_row.copy()
    for i in range(order):
        st = i * order
        ed = (i + 1) * order
        ctmp_col[:, st:ed] = ctmp_row[:, i::order]
    cov = ctmp_col

    return state, cov


def single_point_init(z, R, v_max):
    '''
    Single-point method initializes the state estimate just including position and
    velocity in the sense of Bayesian, which assumes that position components in 
    state to be estimated have infinite prior convariance and velocity components
    have prior convariance (v_max^2) / 3, see[1].
    '''
    zdim = len(z)
    if isinstance(R, (int, float)):
        R = np.diag([R] * zdim)
    if isinstance(v_max, (int, float)):
        v_max = [v_max] * zdim

    state = np.zeros(2 * zdim)
    state[:zdim] = z
    cov = np.zeros((2 * zdim, 2 * zdim))
    cov[:zdim, :zdim] = R
    cov[zdim:, zdim:] = np.diag(v_max)**2 / 3

    return __swap(state, cov, 1)


def two_point_diff_init(z1, z2, R1, R2, T, q=None):
    '''
    Note that The q is an optional process noise parameter. If this parameter is
    None, then it is assumed there is no process noise. This parameter is used in
    discretized continuous-time linear dynamic model
    '''
    zdim = len(z1)
    if isinstance(R1, (int, float)):
        R1 = np.diag([R1] * zdim)
    if isinstance(R2, (int, float)):
        R2 = np.diag([R2] * zdim)
    if q is not None:
        if isinstance(q, (int, float)):
            q = [q] * zdim
    else:
        q = [0] * zdim

    state = np.zeros(2 * zdim)
    state[:zdim] = z2
    state[zdim:] = (z2 - z1) / T
    cov = np.zeros((2 * zdim, 2 * zdim))
    cov[:zdim, :zdim] = R2
    cov[:zdim, zdim:] = cov[zdim:, :zdim] = R2 / T
    cov[zdim:, zdim:] = (R1 + R2) / T**2 + T * np.diag(q) / 3

    return __swap(state, cov, 1)


def biased_three_point_diff_init(z1, z2, z3, R1, R2, R3, T):
    zdim = len(z1)
    if isinstance(R1, (int, float)):
        R1 = np.diag([R1] * zdim)
    if isinstance(R2, (int, float)):
        R2 = np.diag([R2] * zdim)
    if isinstance(R3, (int, float)):
        R3 = np.diag([R3] * zdim)

    state = np.zeros(3 * zdim)
    state[:zdim] = z3
    state[zdim:2 * zdim] = (z3 - z2) / T
    state[2 * zdim:] = (z3 - 2 * z2 + z1) / T**2
    cov = np.zeros((3 * zdim, 3 * zdim))
    cov[:zdim, :zdim] = R3
    cov[:zdim, zdim:2 * zdim] = cov[zdim:2 * zdim, :zdim] = R3 / T
    cov[:zdim, 2 * zdim:] = cov[2 * zdim:, :zdim] = R3 / T**2
    cov[zdim:2 * zdim, zdim:2 * zdim] = (R3 + R2) / T**2
    cov[zdim:2 * zdim, 2 * zdim:] = cov[2 * zdim:, zdim:2 * zdim] = (R3 + 2 * R2) / T**3
    cov[2 * zdim:, 2 * zdim:] = (R3 + 4 * R2 + R1) / T**4

    return __swap(state, cov, 2)


# TODO not sure, it needs careful verification.
def unbiased_three_point_diff_init(z1, z2, z3, R1, R2, R3, T, q=None):
    '''
    Note that The q is an optional process noise parameter. If this parameter is
    None, then it is assumed there is no process noise. This parameter is used in
    discretized continuous-time linear dynamic model
    '''
    zdim = len(z1)
    if isinstance(R1, (int, float)):
        R1 = np.diag([R1] * zdim)
    if isinstance(R2, (int, float)):
        R2 = np.diag([R2] * zdim)
    if isinstance(R3, (int, float)):
        R3 = np.diag([R3] * zdim)
    if q is not None:
        if isinstance(q, (int, float)):
            q = [q] * zdim
    else:
        q = [0] * zdim

    state = np.zeros(3 * zdim)
    state[:zdim] = z3
    state[zdim:2 * zdim] = (3 * z3 - 4 * z2 + z1) / (2 * T)
    state[2 * zdim:] = (z3 - 2 * z2 + z1) / T**2
    cov = np.zeros((3 * zdim, 3 * zdim))
    cov[:zdim, :zdim] = R3
    cov[:zdim, zdim:2 * zdim] = cov[zdim:2 * zdim, :zdim] = 3 * R3 / (2 * T)
    cov[:zdim, 2 * zdim:] = cov[2 * zdim:, :zdim] = R3 / T**2
    cov[zdim: 2 * zdim, zdim: 2 * zdim] = (9 * R3 + 16 * R2 + R1) / (4 * T**2) + 9 * T**3 * np.diag(q) / 80
    cov[zdim: 2 * zdim, 2 * zdim:] = cov[2 * zdim:, zdim: 2 * zdim] = (3 * R3 + 8 * R2 + R1) / (2 * T**3) + T**2 * np.diag(q) / 3
    cov[2 * zdim:, 2 * zdim:] = (R3 + 4 * R2 + R1) / T**4 + 13 * T * np.array(q, dtype=float) / 12

    return __swap(state, cov, 2)

# z1 = np.array([1, 2, 3], dtype=float)
# z2 = np.array([2, 4, 12], dtype=float)
# z3 = np.array([3, 6, 36], dtype=float)
# R1 = R2 = R3 = np.diag([1.2, 1.2, 1.2])
# T = 2
# state, cov = biased_three_point_diff_init(z1, z2, z3, R1, R2, R3, T)
# print(state, cov, sep='\n\n')


# specific initializer
def cv_init(z, R, v_max=None):
    if v_max is None:
        v_var = 100
    else:
        v_var = v_max**2 / 3
    dim = len(z)
    state = np.kron(z, [1, 0])
    cov = np.kron(R, np.diag([1, 0]))
    tmp = np.kron(np.eye(dim), np.diag([0, v_var]))
    cov += tmp

    return state, cov


def ca_init(z, R, v_max=None, a_max=None):
    if v_max is None:
        v_var = 100
    else:
        v_var = v_max**2 / 3
    if a_max is None:
        a_var = 100
    else:
        a_var = a_max**2 / 3
    dim = len(z)
    state = np.kron(z, [1.0, 0.0, 0.0])
    cov = np.kron(R, np.diag([1.0, 0.0, 0.0]))
    tmp = np.kron(np.eye(dim), np.diag([0.0, v_var, a_var]))
    cov += tmp

    return state, cov

# R = np.array([[1.0, 0.2, 0.0], [0.2, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
# z = [1, 3, 0]
# state, cov = ca_init(z, R)
# print(cov)