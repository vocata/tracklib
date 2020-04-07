# -*- coding: utf-8 -*-
'''
This model include Abstract base class named KFBase and PFBase
'''
from __future__ import division, absolute_import, print_function


__all__ = ['KFBase, PFBase']

import abc


class KFBase(abc.ABC):
    def __init__(self):
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
        if self._prior_state is None:
            raise AttributeError("'%s' object has no attribute 'prior_state'" %
                                 self.__class__.__name__)
        return self._prior_state

    @property
    def post_state(self):
        if self._post_state is None:
            raise AttributeError("'%s' object has no attribute 'post_state'" %
                                 self.__class__.__name__)
        return self._post_state

    @property
    def prior_cov(self):
        if self._prior_cov is None:
            raise AttributeError("'%s' object has no attribute 'prior_cov'" %
                                 self.__class__.__name__)
        return self._prior_cov

    @property
    def post_cov(self):
        if self._post_cov is None:
            raise AttributeError("'%s' object has no attribute 'post_cov'" %
                                 self.__class__.__name__)
        return self._post_cov

    @property
    def innov(self):
        if self._innov is None:
            raise AttributeError("'%s' object has no attribute 'innov'" %
                                 self.__class__.__name__)
        return self._innov

    @property
    def innov_cov(self):
        if self._innov_cov is None:
            raise AttributeError("'%s' object has no attribute 'innov_cov'" %
                                 self.__class__.__name__)
        return self._innov_cov

    @property
    def gain(self):
        if self._gain is None:
            raise AttributeError("'%s' object has no attribute 'gain'" %
                                 self.__class__.__name__)
        return self._gain


class PFBase(abc.ABC):
    def __init__(self):
        self._samples = None
        self._weights = None

        self._len = 0
        self._init = False

    def __len__(self):
        return self._len

    @abc.abstractmethod
    def step(self, *args, **kw):
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
    def post_state(self):
        if self.samples is None or self._weights is None:
            raise AttributeError("'%s' object has no attribute 'MMSE'" %
                                 self.__class__.__name__)
        return self._weights @ self._samples
