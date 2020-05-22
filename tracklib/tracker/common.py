# -*- coding: utf-8 -*-
'''
REFERENCES:
'''
from __future__ import division, absolute_import, print_function


__all__ = ['HistoryLogic', 'ScoreLogic', 'Detection']

import numbers
import numpy as np
import tracklib as tlb
from collections.abc import Iterable


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
        if isinstance(meas, np.ndarray):
            self._meas = (meas,)
        elif isinstance(meas, Iterable):
            self._meas = tuple(meas)
        else:
            raise TypeError('meas can not be the type: `%s`' % meas.__class__.__name__)
        if isinstance(cov, np.ndarray):
            self._cov = (cov,)
        elif isinstance(cov, Iterable):
            self._cov = tuple(cov)
        else:
            raise TypeError('cov can not be the type: `%s`' % cov.__class__.__name__)
        if len(meas) != len(cov):
            raise ValueError('the lengths of meas and cov must be the same')
        self._len = len(meas)

    def __iter__(self):
        it = ((self._meas[i], self._cov[i]) for i in range(self._len))
        return it

    def __getitem__(self, n):
        if isinstance(n, numbers.Integral):
            return self._meas[n], self._cov[n]
        elif isinstance(n, Iterable):
            m = [self._meas[i] for i in n]
            c = [self._cov[i] for i in n]
            return m, c
        else:
            raise TypeError('index can not be the type: `%s`' % n.__class__.__name__)

    def __len__(self):
        return self._len
    
    @property
    def meas(self):
        return self._meas

    @property
    def cov(self):
        return self._cov
