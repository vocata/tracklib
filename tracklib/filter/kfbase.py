# -*- coding: utf-8 -*-
'''
This model include Abstract base class named KFBase from which 
all Kalman filter classes are inherited.
'''
from __future__ import division, absolute_import, print_function


__all__ = ['KFBase']

import abc


class KFBase(abc.ABC):
    def __init__(self):
        super().__init__()

        self._prior_state = None
        self._post_state = None
        self._prior_cov = None
        self._post_cov = None
        self._innov = None
        self._innov_cov = None
        self._gain = None

        self._len = 0
        self._stage = 0
        self._init = False

    def __len__(self):
        return self._len

    @abc.abstractmethod
    def predict(self, *args, **kw):
        pass

    @abc.abstractmethod
    def update(self, *args, **kw):
        pass

    @abc.abstractmethod
    def step(self, *args, **kw):
        pass

    @property
    def prior_state(self):
        return self._prior_state

    @property
    def post_state(self):
        return self._post_state

    @property
    def prior_cov(self):
        return self._prior_cov

    @property
    def post_cov(self):
        return self._post_cov

    @property
    def innov(self):
        return self._innov

    @property
    def innov_cov(self):
        return self._innov_cov

    @property
    def gain(self):
        return self._gain
