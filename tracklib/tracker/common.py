# -*- coding: utf-8 -*-
'''
REFERENCES:
'''
from __future__ import division, absolute_import, print_function


__all__ = ['HistoryLogic', 'ScoreLogic', 'Detection']

import numbers
import numpy as np
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
        self._flag[0] = True

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


class ScoreLogic():
    pass


class Detection():
    def __init__(self, meas, cov):
        if isinstance(meas, list):
            self._meas = meas
        else:
            raise ValueError('meas must be a list')
        if isinstance(cov, list):
            self._cov = cov
        else:
            raise ValueError('cov must be a Iterable')
        if len(meas) != len(cov):
            raise ValueError('the lengths of meas and cov must be the same')
        self._len = len(meas)

    def __iter__(self):
        it = ((self._meas[i], self._cov[i]) for i in range(self._len))
        return it

    def __getitem__(self, n):
        if isinstance(n, numbers.Integral):
            return self._meas[n], self._cov[n]
        elif hasattr(n, '__getitem__'):
            m = [self._meas[i] for i in n]
            c = [self._cov[i] for i in n]
            return m, c
        else:
            raise ValueError('index must be a integer, list or tuple')

    def __len__(self):
        return self._len
    
    @property
    def meas(self):
        return self._meas

    @property
    def cov(self):
        return self._cov
