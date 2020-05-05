# -*- coding: utf-8 -*-
'''
REFERENCES:
'''
from __future__ import division, absolute_import, print_function


__all__ = ['HistoryLogic']

import numpy as np
import scipy.optimize as op


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
        self._init = False

    def hit(self):
        self._flag[1:] = self._flag[:-1]
        self._flag[0] = True
        self._init = True

    def miss(self):
        self._flag[1:] = self._flag[:-1]
        self._flag[0] = False
        self._init = True

    def confirm(self):
        if self._init == False:
            return False
        return np.sum(self._flag[:self._c_N] == True) >= self._c_M

    def delete(self):
        if self._init == False:
            return False
        return np.sum(self._flag[:self._d_N] == True) >= self._d_M


class Track():
    def __init__(self, det):
        pass