# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function


__all__ = [
    'is_matrix', 'is_square', 'is_column', 'is_row', 'is_diag', 'is_symmetirc',
    'is_posi_def', 'is_posi_semidef', 'is_neg_def', 'is_neg_semidef', 'col',
    'row', 'deg2rad', 'rad2deg', 'cart2pol', 'pol2cart', 'multi_normal',
    'disc_random'
]

import numpy as np
import scipy.linalg as lg
from collections.abc import Iterable, Iterator

EPSILON = 0.000001


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
    return np.all(e >= 0)


def is_neg_def(x):
    if not is_symmetirc(x):
        return False
    e, _ = lg.eigh(x)
    return np.all(e < 0)


def is_neg_semidef(x):
    if not is_symmetirc(x):
        return False
    e, _ = lg.eigh(x)
    return np.all(e <= 0)


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
        raise TypeError('parametes must be real number or iterable, not %s', x.__class__.__name__)
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
        raise TypeError('parametes must be real number or iterable, not %s', x.__class__.__name__)
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


def multi_normal(mean, cov, Ns=1, axis=0):
    '''
    Draw random samples from a normal (Gaussian) distribution with mean and cov

    Parameters
    ----------
    mean : 1-D array_like, of length N or int
        Mean of the N-dimensional distribution.
    cov : 2-D array_like, of shape (N, N)
        Covariance matrix of the distribution. It must be symmetric and
        positive-semidefinite for proper sampling.
    Ns : int, optional
        Number of samples. Default is 0
    axis : int, optional
        The axis along which the noise will be generated. Default is 0

    Returns
    -------
    out : ndarray
        The drawn samples of shape (Ns, N) if axis is 0 or (N, Ns) axis is 1
    '''
    if not is_posi_semidef(cov):
        raise ValueError('convariance matrix must be posotive semi-definite')

    N = cov.shape[0]
    if isinstance(mean, int):
        mean = np.zeros(N) + mean
    U, S, V = lg.svd(cov)
    D = U @ np.diag(np.sqrt(S)) @ V.T
    if Ns == 1:
        wgn = np.random.randn(N)
    else:
        wgn = np.random.randn(N, Ns)
    if axis == 0:
        out = np.dot(wgn.T, D.T)
        out += mean
    elif axis == 1:
        out = np.dot(D, wgn)
        out += np.reshape(mean, (-1, 1))
    else:
        raise ValueError('axis must be 0 or 1')
    return out


def disc_random(prob, Ns=1, scope=None, alg='roulette'):
    '''
    Draw random samples from a discrete distribution

    Parameters
    ----------
    prob : list, of length N
        Discrete probability
    Ns : int, optional
        Number of samples
    scope : list, optional
        The scope in which the samples will be drawn. Default is 0
    alg : str, optional
        Sample algorithm, it can be 'roulette' and 'low_var'

    Returns
    -------
    rv : list
        The drawn samples from scope
    index : list
        The index corresponding to the sample drawn from the scope
    '''
    rv_num = len(prob)
    if scope is None:
        scope = np.arange(rv_num)

    rv = []
    index = []

    if alg == 'roulette':
        cdf = list(range(rv_num + 1))
        for i in range(rv_num):
            cdf[i + 1] = cdf[i] + prob[i]
        for i in range(Ns):
            idx = 0
            rnd = np.random.rand()
            while cdf[idx] < rnd:
                idx += 1
            rv.append(scope[idx - 1])
            index.append(idx - 1)
    elif alg == 'low_var':
        rnd = np.random.rand() / Ns
        cdf = prob[0]
        idx = 0
        for i in range(Ns):
            u = rnd + i / Ns
            while u > cdf:
                idx += 1
                cdf += prob[idx]
            rv.append(scope[idx])
            index.append(idx)
    else:
        raise ValueError("alg must be 'roulette' or 'low_var'")

    return rv, index
