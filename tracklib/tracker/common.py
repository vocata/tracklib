# -*- coding: utf-8 -*-
'''
REFERENCES:
'''
from __future__ import division, absolute_import, print_function


__all__ = ['HistoryLogic', 'Track', 'Detection']

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


class Track():
    track_id = 0
    def __init__(self, filter, logic):
        self._filter = filter
        self._logic = logic
        self._is_coasted = False
        self._age = 1
        self._id = Track.track_id

        Track.track_id += 1
    
    def distance(self, target):
        if self._is_coasted:
            pass
        else:
            pass

    def assign(self, target, cov):
        self._filter.update(target, R=cov)
        if self._logic.type() == 'history':
            self._logic.hit()
        else:
            pass    # TODO other logic
        self._is_coasted = False
        self._age += 1

    def coast(self):
        self._filter.skin()
        if self._logic.type() == 'history':
            self._logic.miss()
        else:
            pass    # TODO other logic
        self._is_coasted = True
        self._age += 1

    def is_confirmed(self):
        return self._logic.confirmed()
    
    def is_detached(self):
        return self._logic.detached()

    def age(self):
        return self._age

    def id(self):
        return self._id


class Detection():
    def __init__(self, data, covariance, coordinate):
        '''
        coordinate == cartesian: data[0], data[1], data[2] == x, y, z.
        coordinate == polar: data[0], data[1], data[2] == range, azimuth, elevation
        '''
        dim = len(data)
        if coordinate == 'cartesian':
            self._data = data
            self._covariance = covariance
        elif coordinate == 'polar':
            if dim == 2:
                self._data = np.array([tlb.pol2cart(*d) for d in data], dtype=float)
                # TODO coverted measurement covariance
                # self.covariance == ?
            if dim == 3:
                self.data = np.array([tlb.sph2cart(*d) for d in data], dtype=float)
                # TODO coverted measurement covariance
                # self.covariance == ?
        else:
            raise ValueError('unknown coordinate: %s' % coordinate)
        
    def __iter__(self):
        iter(self._data)
    
    def covariance(self):
        return self._covariance