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
import tracklib.model as model
from .base import KFBase


class MMFilter(KFBase):
    '''
    Static multiple model filter
    '''
    def __init__(self, switch_fcn=model.model_switch):
        super().__init__()
        self._switch_fcn = switch_fcn

        self._models = []
        self._model_types = []
        self._probs = None
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
        if n < 0 or n >= self._models_n:
            raise IndexError('index out of range')
        return self._models[n], self._probs[n]

    def __update(self):
        state_org = [self._models[i].state for i in range(self._models_n)]
        cov_org = [self._models[i].cov for i in range(self._models_n)]
        types = [self._model_types[i] for i in range(self._models_n)]

        xtmp = 0
        for i in range(self._models_n):
            xi = self._switch_fcn(state_org[i], types[i], types[0])
            xtmp += self._probs[i] * xi
        self._state = xtmp

        Ptmp = 0
        for i in range(self._models_n):
            xi = self._switch_fcn(state_org[i], types[i], types[0])
            Pi = self._switch_fcn(cov_org[i], types[i], types[0])
            err = xi - xtmp
            Ptmp += self._probs[i] * (Pi + np.outer(err, err))
        Ptmp = (Ptmp + Ptmp.T) / 2
        self._cov = Ptmp

    def _set_state(self, state):
        if self._models_n == 0:
            raise AttributeError("AttributeError: can't set attribute")
        for i in range(self._models_n):
            xi = self._switch_fcn(state, self._model_types[0], self._model_types[i])
            self._models[i].state = xi
    
    def _set_cov(self, cov):
        if self._models_n == 0:
            raise AttributeError("AttributeError: can't set attribute")
        for i in range(self._models_n):
            Pi = self._switch_fcn(cov, self._model_types[0], self._model_types[i])
            self._models[i].cov = Pi
    
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
        if isinstance(state, np.ndarray):
            state = [state] * self._models_n
        if isinstance(cov, np.ndarray):
            cov = [cov] * self._models_n

        for i in range(self._models_n):
            x = self._switch_fcn(state[i], self._model_types[0], self._model_types[i])
            P = self._switch_fcn(cov[i], self._model_types[0], self._model_types[i])
            self._models[i].init(x, P)
        self.__update()
        self._init = True

    def add_models(self, models, model_types, probs=None):
        '''
        Add new model

        Parameters
        ----------
        models : list, of length N
            the list of Kalman filter
        model_types : list, of length N
            the types corresponding to models
        probs : 1-D array_like, of length N
            model probability

        Returns
        -------
            None
        '''
        self._models_n = len(models)
        self._models.extend(models)
        self._model_types.extend(model_types)
        if probs is None:
            self._probs = np.ones(self._models_n) / self._models_n
        else:
            self._probs = np.copy(probs)

    def predict(self, u=None, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        for i in range(self._models_n):
            self._models[i].predict(u, **kwargs)
        # update prior state and covariance
        self.__update()

    def correct(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        # update probability
        pdf = np.zeros(self._models_n)
        for i in range(self._models_n):
            pdf[i] = self._models[i].likelihood(z, **kwargs)
            self._models[i].correct(z, **kwargs)
        # update model probability
        self._probs *= pdf
        self._probs /= np.sum(self._probs)
        # update posterior state and covariance
        self.__update()

    def distance(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        d = 0
        for i in range(self._models_n):
            d += self._probs[i] * self._models[i].distance(z, **kwargs)
        
        return d

    def likelihood(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        pdf = 0
        for i in range(self._models_n):
            pdf += self._probs[i] * self._models[i].likelihood(z, **kwargs)
        
        return pdf

    def models(self):
        return self._models

    def probs(self):
        return self._probs
