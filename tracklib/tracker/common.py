# -*- coding: utf-8 -*-
'''
REFERENCES:
'''
from __future__ import division, absolute_import, print_function


__all__ = ['HistoryLogic', 'Detection']

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
        return np.sum(self._flag[:self._d_N] == True) >= self._d_M

    def type(self):
        return self._logic_type


class ScoreLogic():
    pass


class Detection():
    def __init__(self, data, covariance, coordinate):
        self._len = data.shape[0]
        self._dim = data.shape[1]
        if coordinate == 'cartesian':
            self._data = data
            self._covariance = covariance
        elif coordinate == 'polar':
            if self._dim == 2:
                self._data = np.array([tlb.pol2cart(*d) for d in data], dtype=float)
                # TODO coverted measurement covariance
                # self.covariance == ?
            if self._dim == 3:
                self.data = np.array([tlb.sph2cart(*d) for d in data], dtype=float)
                # TODO coverted measurement covariance
                # self.covariance == ?
        else:
            raise ValueError('unknown coordinate: %s' % coordinate)
        
    def __iter__(self):
        iter(self._data)

    def __getitem__(self, n):
        if n < 0 or n >= self._len:
            raise IndexError('index out of range')
        return self._data[n]
    
    def __len__(self):
        return self._len
    
    def covariance(self):
        return self._covariance