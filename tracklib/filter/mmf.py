# -*- coding: utf-8 -*-
'''
The static multiple model filter can use other types of Kalman filters as its submodels
for filtering. Currently supported filters are stardard Kalman filter, extended Kalman
filter and unscented Kalman filter. For the non-linear system with additive Gaussian noise,
this multiple model filter can be used as Gaussian sum filter which by setting different initial
state and convariance of each non-linear filters or submodels and viewing model probability
as weight of each Gaussian density constituting the Gaussian mixture.

[1]. D. Simon, "Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches," John Wiley and Sons, Inc., 2006.
[2]. Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation: Theory, Algorithms and Software," New York: Wiley, 2001
'''
from __future__ import division, absolute_import, print_function


__all__ = ['MMFilter']

import numpy as np
import scipy.linalg as lg
from .base import KFBase


class MMFilter(KFBase):
    '''
    Static multiple model filter
    '''
    def __init__(self):
        super().__init__()
        self._models = []
        self._probs = []
        self._models_n = 0

    def __str__(self):
        msg = 'Static multiple model filter:\n{\n  '
        if self._models_n < 10:
            sub = ['{}: model: {}, probability: {}'.format(i, self._models[i], self._probs[i]) for i in range(self._models_n)]
            sub = '\n  '.join(sub)
        else:
            sub = ['{}: model: {}, probability: {}'.format(i, self._models[i], self._probs[i]) for i in range(3)]
            sub.append('...')
            sub.extend(['{}: model: {}, probability: {}'.format(i, self._models[i], self._probs[i]) for i in range(self._models_n - 3, self._models_n)])
            sub = '\n  '.join(sub)
        msg += sub
        msg += '\n}'
        return msg

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return ((self._models[i], self._probs[i]) for i in range(self._models_n))

    def __getitem__(self, n):
        return self._models[n], self._probs[n]

    def init(self, state, cov):
        '''
        Initial filter

        Parameters
        ----------
        state : ndarray or list
            Initial prior state estimate
        cov : ndarray or list
            Initial error convariance matrix

        Returns
        -------
            None
        '''
        if self._models_n == 0:
            raise RuntimeError('models must be added before calling init')
        if isinstance(state, list):
            pass
        elif isinstance(state, np.ndarray):
            state = [state] * self._models_n
        else:
            raise TypeError('state must be a ndarray, list, not %s' % state.__class__.__name__)
        if isinstance(cov, list):
            pass
        elif isinstance(cov, np.ndarray):
            cov = [cov] * self._models_n
        else:
            raise TypeError('cov must be a ndarray, list, not %s' % cov.__class__.__name__)

        for i in range(self._models_n):
            self._models[i].init(state[i], cov[i])
        self._len = 0
        self._stage = 0
        self._init = True

    def add_models(self, models, probs):
        '''
        Add new model

        Parameters
        ----------
        models : list
            the list of Kalman filter
        probs : list
            model prior probability

        Returns
        -------
            None
        '''
        if not isinstance(models, list):
            raise TypeError('models must be a list, not %s' %
                            models.__class__.__name__)
        if not isinstance(probs, list):
            raise TypeError('probs must be a list, not %s' %
                            probs.__class__.__name__)
        if len(models) != len(probs):
            raise ValueError('the length of models must be the same as probs')

        self._models.extend(models)
        self._probs.extend(probs)
        self._models_n = len(models)

    def predict(self, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        for i in range(self._models_n):
            self._models[i].predict(u, **kw)

        self._stage = 1

    def update(self, z, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        # update probability
        pdf = np.zeros(self._models_n)
        for i in range(self._models_n):
            self._models[i].update(z, **kw)
            r = self._models[i].innov
            S = self._models[i].innov_cov
            # If there is a singular value, exp will be very small and all values in the pdf will be 0,
            # then total defined below will be 0 and an ZeroDivisionError will occur.
            pdf[i] = np.exp(-r @ lg.inv(S) @ r / 2) / np.sqrt(lg.det(2 * np.pi * S))
            self._probs[i] *= pdf[i]
        # normalize
        self._probs[:] /= np.sum(self._probs)

        self._len += 1
        self._stage = 0

    def step(self, z, u=None, **kw):
        assert (self._stage == 0)

        self.predict(u, **kw)
        self.update(z, **kw)

    @property
    def weighted_state(self):
        # weighted state estimate
        if self._models_n == 0:
            raise AttributeError("'%s' object has no attribute 'weighted_state'" %
                                 self.__class__.__name__)
        state = 0
        for i in range(self._models_n):
            state += self._probs[i] * self._models[i].post_state
        return state

    @property
    def maxprob_state(self):
        # state estimate of models with maximum probability
        if self._models_n == 0:
            raise AttributeError("'%s' object has no attribute 'maxprob_state'" %
                                 self.__class__.__name__)
        return self._models[np.argmax(self._probs)].post_state

    @property
    def models(self):
        if self._models_n == 0:
            raise AttributeError("'%s' object has no attribute 'models'" %
                                 self.__class__.__name__)
        return self._models

    @property
    def probs(self):
        if self._models_n == 0:
            raise AttributeError("'%s' object has no attribute 'probs'" %
                                 self.__class__.__name__)
        return self._probs

    @property
    def post_state(self):
        if self._models_n == 0:
            raise AttributeError("'%s' object has no attribute 'post_state'" %
                                 self.__class__.__name__)
        return self.weighted_state
    
    @post_state.setter
    def post_state(self, state):
        if self._models_n == 0:
            raise AttributeError("'%s' object has no attribute 'post_state'" %
                                 self.__class__.__name__)
        for i in range(self._models_n):
            self._models[i].post_state = state
    
    @property
    def post_cov(self):
        if self._models_n == 0:
            raise AttributeError("'%s' object has no attribute 'post_cov'" %
                                 self.__class__.__name__)
        post_state = self.post_state
        post_cov = 0
        for i in range(self._models_n):
            err = self._models[i].post_state - post_state
            post_cov += self._probs[i] * (self._models[i].post_cov + np.outer(err, err))
        return (post_cov + post_cov.T) / 2

    @post_cov.setter
    def post_cov(self, cov):
        if self._models_n == 0:
            raise AttributeError("'%s' object has no attribute 'post_cov'" %
                                 self.__class__.__name__)
        for i in range(self._models_n):
            self._models_n[i].post_cov = cov
