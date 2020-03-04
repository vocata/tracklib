# -*- coding: utf-8 -*-

import numpy as np
from collections.abc import Iterable, Iterator

__all__ = [
    'deg2rad', 'rad2deg', 'cart2pol', 'pol2cart', 'is_matrix', 'is_square',
    'is_column', 'is_row', 'is_diag', 'col', 'row'
]


def deg2rad(deg):
    rad = np.pi / 180 * deg
    return rad


def rad2deg(rad):
    deg = 180 / np.pi * rad
    return deg


def cart2pol(x, y, z=None):
    '''
    Transform Cartesian to polar coordinates or cylindrical coordinates
    '''

    r = (x**2 + y**2)**(1 / 2)
    th = np.arctan2(y, x)

    return (r, th, z) if z else (r, th)


def pol2cart(r, th, z=None):
    '''
    Transform polar coordinates or cylindrical coordinates to Cartesian
    '''

    x = r * np.cos(th)
    y = r * np.sin(th)

    return (x, y, z) if z else (x, y)


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


def col(x, *args, dtype='float64', **kw):
    '''
    Converts numbers or iterable objects to column vectors
    and sets the data type to 'float64' default.
    '''

    isnum = lambda x: (isinstance(x, int) or isinstance(x, float))
    if isnum(x):
        x = np.array([x], *args, dtype=dtype, **kw).reshape((-1, 1))
    elif isinstance(x, Iterator):
        x = np.array(list(x), *args, dtype=dtype, **kw).reshape((-1, 1))
    elif isinstance(x, Iterable):
        x = np.array(x, *args, dtype=dtype, **kw).reshape((-1, 1))
    else:
        raise ValueError('parametes must be number or iterable')
    return x


def row(x, *args, dtype='float64', **kw):
    '''
    Converts numbers or iterable objects to row vectors
    and sets the data type to 'float64' default.
    '''

    isnum = lambda x: (isinstance(x, int) or isinstance(x, float))
    if isnum(x):
        x = np.array([x], *args, dtype=dtype, **kw).reshape((1, -1))
    elif isinstance(x, Iterator):
        x = np.array(list(x), *args, dtype=dtype, **kw).reshape((1, -1))
    elif isinstance(x, Iterable):
        x = np.array(x, *args, dtype=dtype, **kw).reshape((1, -1))
    else:
        raise ValueError('parametes must be number or iterable')
    return x