# -*- coding: utf-8 -*-
'''
This model include Abstract base class named KFBase and PFBase
'''
from __future__ import division, absolute_import, print_function


__all__ = ['KFBase, PFBase']

import abc


class KFBase(abc.ABC):
    def __init__(self):
        self._state = None
        self._cov = None

    @abc.abstractmethod
    def init(self, state, cov):
        pass

    @abc.abstractmethod
    def reset(self, state, cov):
        pass

    @abc.abstractmethod
    def predict(self, u=None, **kwargs):
        pass

    @abc.abstractmethod
    def correct(self, z, **kwargs):
        pass

    @abc.abstractmethod
    def distance(self, z, **kwargs):
        pass

    @abc.abstractmethod
    def likelihood(self, z, **kwargs):
        pass

    @property
    def state(self):
        if self._state is not None:
            return self._state.copy()
        else:
            raise AttributeError("'%s' object has no attribute 'state'" %
                                 self.__class__.__name__)

    @property
    def cov(self):
        if self._cov is not None:
            return self._cov.copy()
        else:
            raise AttributeError("'%s' object has no attribute 'cov'" %
                                 self.__class__.__name__)
    

class PFBase(abc.ABC):
    def __init__(self):
        self._samples = None
        self._weights = None

        self._len = 0
        self._init = False

    def __len__(self):
        return self._len

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        pass
    
    @property
    def samples(self):
        if self._samples is None:
            raise AttributeError("'%s' object has no attribute 'samples'" %
                                 self.__class__.__name__)
        return self._samples
        
    @property
    def weights(self):
        if self._weights is None:
            raise AttributeError("'%s' object has no attribute 'weights'" %
                                 self.__class__.__name__)
        return self._weights

    @property
    def MMSE(self):
        if self.samples is None or self._weights is None:
            raise AttributeError("'%s' object has no attribute 'MMSE'" %
                                 self.__class__.__name__)
        return self._weights @ self._samples

    # not a good estimate for particle filter
    @property
    def MAP(self):
        if self.samples is None or self._weights is None:
            raise AttributeError("'%s' object has no attribute 'MAP'" %
                                 self.__class__.__name__)
        return self._samples[self._weights.argmax()]
