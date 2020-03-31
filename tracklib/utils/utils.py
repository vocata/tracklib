# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function


__all__ = [
    'is_matrix', 'is_square', 'is_column', 'is_row', 'is_diag', 'is_symmetirc',
    'is_posi_def', 'is_posi_semidef', 'is_neg_def', 'is_neg_semidef', 'col',
    'row', 'deg2rad', 'rad2deg', 'cart2pol', 'pol2cart', 'crandn'
]

import numpy as np
import scipy.linalg as lg
from collections.abc import Iterable, Iterator


def is_matrix(x):
    return isinstance(x, np.ndarray) and len(x.shape) == 2


def is_square(x):
    return is_matrix(x) and x.shape[0] == x.shape[1]


def is_column(x):
    return is_matrix(x) and x.shape[1] == 1


def is_row(x):
    return is_matrix(x) and x.shape[0] == 1


def is_diag(x):
    return is_matrix(x) and (x == np.diag(x.diagonal())).all()


def is_symmetirc(x):
    return not np.any(x - x.T)


def is_posi_def(x):
    if not is_symmetirc(x):
        return False
    e, _ = lg.eigh(x)
    return np.all(e > 0)


def is_posi_semidef(x):
    if not is_symmetirc(x):
        return False
    e, _ = lg.eigh(x)
    return np.all(e >= 0) and np.any(e == 0)


def is_neg_def(x):
    if not is_symmetirc(x):
        return False
    e, _ = lg.eigh(x)
    return np.all(e < 0)


def is_neg_semidef(x):
    if not is_symmetirc(x):
        return False
    e, _ = lg.eigh(x)
    return np.all(e <= 0) and np.any(e == 0)


def col(x, *args, dtype=float, **kw):
    '''
    Converts numbers or iterable objects to column vectors
    and sets the data type to 'float' default.
    '''
    if isinstance(x, int) or isinstance(x, float):
        x = np.array([x], *args, dtype=dtype, **kw).reshape((-1, 1))
    elif isinstance(x, Iterator):
        x = np.array(list(x), *args, dtype=dtype, **kw).reshape((-1, 1))
    elif isinstance(x, Iterable):
        x = np.array(x, *args, dtype=dtype, **kw).reshape((-1, 1))
    else:
        raise ValueError('parametes must be number or iterable')
    return x


def row(x, *args, dtype=float, **kw):
    '''
    Converts numbers or iterable objects to row vectors
    and sets the data type to 'float' default.
    '''
    if isinstance(x, int) or isinstance(x, float):
        x = np.array([x], *args, dtype=dtype, **kw).reshape((1, -1))
    elif isinstance(x, Iterator):
        x = np.array(list(x), *args, dtype=dtype, **kw).reshape((1, -1))
    elif isinstance(x, Iterable):
        x = np.array(x, *args, dtype=dtype, **kw).reshape((1, -1))
    else:
        raise ValueError('parametes must be number or iterable')
    return x


def deg2rad(deg):
    rad = np.pi / 180 * deg
    return rad


def rad2deg(rad):
    deg = 180 / np.pi * rad
    return deg


def cart2pol(x, y, z=None):
    r = (x**2 + y**2)**(1 / 2)
    th = np.arctan2(y, x)

    return (r, th, z) if z else (r, th)


def pol2cart(r, th, z=None):
    x = r * np.cos(th)
    y = r * np.sin(th)

    return (x, y, z) if z else (x, y)


def crandn(cov, N=1):
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
    if np.any(e < 0):
        raise ValueError('convariance matrix must be posotive definite')
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