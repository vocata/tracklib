# -*- coding: utf-8 -*-
'''
This model include Abstract base class named KFBase and PFBase
'''
from __future__ import division, absolute_import, print_function


__all__ = ['KFBase, PFBase']

import abc


class KFBase(abc.ABC):
    def __init__(self):
        self._xdim = None
        self._wdim = None
        self._zdim = None
        self._vdim = None
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

    # @abc.abstractmethod
    # def distance(self, z):
    #     pass

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
    def xdim(self):
        if self._xdim is None:
            raise AttributeError("'%s' object has no attribute 'xdim'" %
                                 self.__class__.__name__)
        return self._xdim
    
    @property
    def wdim(self):
        if self._wdim is None:
            raise AttributeError("'%s' object has no attribute 'wdim'" %
                                 self.__class__.__name__)
        return self._wdim

    @property
    def zdim(self):
        if self._zdim is None:
            raise AttributeError("'%s' object has no attribute 'zdim'" %
                                 self.__class__.__name__)
        return self._zdim
    
    @property
    def vdim(self):
        if self._vdim is None:
            raise AttributeError("'%s' object has no attribute 'vdim'" %
                                 self.__class__.__name__)
        return self._vdim

    @property
    def prior_state(self):
        if hasattr(self, '_get_prior_state'):
            return self._get_prior_state()
        elif self._prior_state is not None:
            return self._prior_state
        else:
            raise AttributeError("'%s' object has no attribute 'prior_state'" %
                                 self.__class__.__name__)
    
    @prior_state.setter
    def prior_state(self, state):
        if hasattr(self, '_set_prior_state'):
            self._set_prior_state(state)
        else:
            raise AttributeError("AttributeError: can't set attribute")

    @property
    def prior_cov(self):
        if hasattr(self, '_get_prior_cov'):
            return self._get_prior_cov()
        elif self._prior_cov is not None:
            return self._prior_cov
        else:
            raise AttributeError("'%s' object has no attribute 'prior_cov'" %
                                 self.__class__.__name__)
    
    @prior_cov.setter
    def prior_cov(self, cov):
        if hasattr(self, '_set_prior_cov'):
            self._set_post_state(cov)
        else:
            raise AttributeError("AttributeError: can't set attribute")

    @property
    def post_state(self):
        if hasattr(self, '_get_post_state'):
            return self._get_post_state()
        elif self._post_state is not None:
            return self._post_state
        else:
            raise AttributeError("'%s' object has no attribute 'post_state'" %
                                 self.__class__.__name__)
    
    @post_state.setter
    def post_state(self, state):
        if hasattr(self, '_set_post_state'):
            self._set_post_state(state)
        else:
            raise AttributeError("AttributeError: can't set attribute")

    @property
    def post_cov(self):
        if hasattr(self, '_get_post_cov'):
            return self._get_post_cov()
        elif self._post_cov is not None:
            return self._post_cov
        else:
            raise AttributeError("'%s' object has no attribute 'post_cov'" %
                                 self.__class__.__name__)

    @post_cov.setter
    def post_cov(self, cov):
        if hasattr(self, '_set_post_cov'):
            self._set_post_cov(cov)
        else:
            raise AttributeError("AttributeError: can't set attribute")

    @property
    def innov(self):
        if hasattr(self, '_get_innov'):
            return self._get_innov()
        elif self._innov is not None:
            return self._innov
        else:
            raise AttributeError("'%s' object has no attribute 'innov'" %
                                 self.__class__.__name__)

    @innov.setter
    def innov(self, inno):
        if hasattr(self, '_set_innov'):
            self._set_innov(inno)
        else:
            raise AttributeError("AttributeError: can't set attribute")

    @property
    def innov_cov(self):
        if hasattr(self, '_get_innov_cov'):
            return self._get_innov_cov()
        elif self._innov_cov is not None:
            return self._innov_cov
        else:
            raise AttributeError("'%s' object has no attribute 'innov_cov'" %
                                 self.__class__.__name__)

    @innov_cov.setter
    def innov_cov(self, cov):
        if hasattr(self, '_set_innov_cov'):
            self._set_innov_cov(cov)
        else:
            raise AttributeError("AttributeError: can't set attribute")

    @property
    def gain(self):
        if hasattr(self, '_get_gain'):
            return self._get_gain()
        elif self._gain is not None:
            return self._gain
        else:
            raise AttributeError("'%s' object has no attribute 'gain'" %
                                 self.__class__.__name__)

    @gain.setter
    def gain(self, g):
        if hasattr(self, '_set_gain'):
            self._set_gain(g)
        else:
            raise AttributeError("AttributeError: can't set attribute")


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
