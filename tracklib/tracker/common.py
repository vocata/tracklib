# -*- coding: utf-8 -*-
'''
REFERENCES:
'''
from __future__ import division, absolute_import, print_function


__all__ = ['HistoryLogic', 'ScoreLogic', 'Detection']

import numpy as np
import scipy.optimize as op
import tracklib as tlb


def bin_count(x):
    '''
    hamming weight
    '''
    x = int(x)
    if x < 0:
        raise ValueError('x must be a positive integer')
    sum = 0
    while x > 0:
        sum += x & 1
        x = x >> 1
    return sum


class HistoryLogic():
    def __init__(self, confirm_M=2, confirm_N=3, delete_M=5, delete_N=5):
        self._c_M = confirm_M
        self._c_N= confirm_N
        self._d_M = delete_M
        self._d_N = delete_N
        max_N = max(confirm_N, delete_N)
        self._flag = np.zeros(max_N, dtype=np.bool)
        self._flag[0] = 1
        self._logic_type = 'history'

    def hit(self):
        self._flag[1:] = self._flag[:-1]
        self._flag[0] = True

    def miss(self):
        self._flag[1:] = self._flag[:-1]
        self._flag[0] = False

    def confirmed(self):
        return np.sum(self._flag[:self._c_N] == True) >= self._c_M

    def detached(self):
        return np.sum(self._flag[:self._d_N] == False) >= self._d_M

    def type(self):
        return self._logic_type


class ScoreLogic():
    pass


class Detection():
    def __init__(self, data=None, covariance=None):
        if data is None:
            self._data = []
            self._cov = []
        else:
            self._data = data
            self._cov = covariance
        self._len = len(self._data)

    def __iter__(self):
        it = ((self._data[i], self._cov[i]) for i in range(self._len))
        return it

    def __getitem__(self, n):
        if n < 0 or n >= self._len:
            raise IndexError('index out of range')
        return self._data[n], self._cov[n]

    def __len__(self):
        return self._len
    
    @property
    def data(self):
        return self._data

    @property
    def covariance(self):
        return self._cov
