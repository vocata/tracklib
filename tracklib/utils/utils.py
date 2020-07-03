# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function


__all__ = [
    'is_matrix', 'is_square', 'is_column', 'is_row', 'is_diag', 'is_symmetirc',
    'col', 'row', 'deg2rad', 'rad2deg', 'cart2pol', 'pol2cart', 'cart2sph',
    'sph2cart', 'ellipsoidal_volume', 'cholcov', 'multi_normal', 'disc_random',
    'Scope', 'Pair'
]

import numbers
import numpy as np
import scipy.linalg as lg
import scipy.special as sl
from collections.abc import Iterable


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


def col(x, *args, dtype=float, **kw):
    '''
    Converts numbers or iterable objects to column vectors
    and sets the data type to 'float' default.
    '''
    if isinstance(x, numbers.Number):
        x = np.array([x], *args, dtype=dtype, **kw).reshape((-1, 1))
    elif isinstance(x, Iterable):
        x = np.array(tuple(x), *args, dtype=dtype, **kw).reshape((-1, 1))
    else:
        raise TypeError("error 'x' type: '%s'" % x.__class__.__name__)
    return x


def row(x, *args, dtype=float, **kw):
    '''
    Converts numbers or iterable objects to row vectors
    and sets the data type to 'float' default.
    '''
    if isinstance(x, numbers.Number):
        x = np.array([x], *args, dtype=dtype, **kw).reshape((1, -1))
    elif isinstance(x, Iterable):
        x = np.array(tuple(x), *args, dtype=dtype, **kw).reshape((1, -1))
    else:
        raise TypeError("error 'x' type: '%s'" % x.__class__.__name__)
    return x


def deg2rad(deg):
    rad = np.pi / 180 * deg
    return rad


def rad2deg(rad):
    deg = 180 / np.pi * rad
    return deg


def cart2pol(x, y, z=None):
    r = np.sqrt(x**2 + y**2)
    az = np.arctan2(y, x)

    return (r, az, z) if z else (r, az)


def pol2cart(r, az, z=None):
    x = r * np.cos(az)
    y = r * np.sin(az)

    return (x, y, z) if z else (x, y)


def cart2sph(x, y, z):
    proj = np.sqrt(x**2 + y**2)
    r = np.sqrt(proj**2 + z**2)
    az = np.arctan2(y, x)
    elev = np.arctan2(z, proj)

    return r, az, elev


def sph2cart(r, az, elev):
    z = r * np.sin(elev)
    proj = r * np.cos(elev)
    x = proj * np.cos(az)
    y = proj * np.sin(az)

    return x, y, z


def ellipsoidal_volume(X):
    n = X.shape[0] / 2
    vol = np.pi**n * np.sqrt(lg.det(X)) / sl.gamma(n + 1)
    return vol


def cholcov(cov, lower=False):
    '''
    Compute the Cholesky-like decomposition of a matrix.

    return S such that cov = dot(S.T,S). `cov` must be square, symmetric, and
    positive semi-definite. If `cov` is positive definite, then S is the square,
    upper triangular Cholesky factor. If 'cov' is not positive definite, S is
    computed from an eigenvalue decomposition of cov. S is not necessarily triangular.

    Parameters
    ----------
    cov : 2-D array_like, of shape (N, N)
        Matrix to be decomposed
    lower : bool, optional
        Whether to compute the upper or lower triangular Cholesky
        factorization.  Default is upper-triangular.

    Returns
    -------
    S : (N, N) ndarray
        Upper- or lower-triangular Cholesky factor of `cov`.
    '''
    try:
        S = lg.cholesky(cov, lower)
    except lg.LinAlgError:
        U, s, V = lg.svd(cov)
        if lower:
            S = U @ np.diag(np.sqrt(s)) @ V.T
        else:
            S = U.T @ np.diag(np.sqrt(s)) @ V
    return S


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
        The drawn samples of shape (Ns, N) if axis is 0, (N, Ns) axis is 1
    '''

    dim = cov.shape[0]
    if isinstance(mean, numbers.Number):
        mean = np.full(dim, mean, dtype=float)
    D = cholcov(cov, lower=True)
    if Ns == 1:
        wgn = np.random.randn(dim)
    else:
        wgn = np.random.randn(dim, Ns)
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
        raise ValueError('unknown algorithem: %s' % alg)

    return rv, index


class Scope():
    def __init__(self, value_min, value_max):
        assert (value_min <= value_max)
        self._min = value_min
        self._max = value_max

    def within(self, value):
        return self._min <= value <= self._max


class Pair(tuple):
    def __new__(cls, value1, value2):
        return super().__new__(cls, (value1, value2))
