# -*- coding: utf-8 -*-
'''
This model include Abstract base class
'''
from __future__ import division, absolute_import, print_function


__all__ = ['FilterBase', 'EOFilterBase']

import abc


class FilterBase(abc.ABC):
    def __init__(self):
        self._state = None
        self._cov = None
        self._init = False

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
    

class EOFilterBase(abc.ABC):
    def __init__(self):
        self._state = None
        self._cov = None
        self._ext = None

    @abc.abstractmethod
    def init(self, state, cov, df, extension):
        pass

    @abc.abstractmethod
    def predict(self, u=None, **kwargs):
        pass

    @abc.abstractmethod
    def correct(self, zs, **kwargs):
        pass

    @abc.abstractmethod
    def distance(self, zs, **kwargs):
        pass

    @abc.abstractmethod
    def likelihood(self, zs, **kwargs):
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
    
    @property
    def extension(self):
        if self._cov is not None:
            return self._ext.copy()
        else:
            raise AttributeError("'%s' object has no attribute 'extension'" %
                                 self.__class__.__name__)
