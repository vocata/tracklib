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

    def __prior_update(self):
        state_org = [self._models[i].prior_state for i in range(self._models_n)]
        cov_org = [self._models[i].prior_cov for i in range(self._models_n)]
        types = [self._model_types[i] for i in range(self._models_n)]

        xtmp = 0
        for i in range(self._models_n):
            xi = self._switch_fcn(state_org[i], types[i], types[0])
            xtmp += self._probs[i] * xi
        self._prior_state = xtmp

        Ptmp = 0
        for i in range(self._models_n):
            xi = self._switch_fcn(state_org[i], types[i], types[0])
            Pi = self._switch_fcn(cov_org[i], types[i], types[0])
            err = xi - xtmp
            Ptmp += self._probs[i] * (Pi + np.outer(err, err))
        Ptmp = (Ptmp + Ptmp.T) / 2
        self._prior_cov = Ptmp

    def __post_update(self):
        state_org = [self._models[i].post_state for i in range(self._models_n)]
        cov_org = [self._models[i].post_cov for i in range(self._models_n)]
        types = [self._model_types[i] for i in range(self._models_n)]

        xtmp = 0
        for i in range(self._models_n):
            xi = self._switch_fcn(state_org[i], types[i], types[0])
            xtmp += self._probs[i] * xi
        self._post_state = xtmp

        Ptmp = 0
        for i in range(self._models_n):
            xi = self._switch_fcn(state_org[i], types[i], types[0])
            Pi = self._switch_fcn(cov_org[i], types[i], types[0])
            err = xi - xtmp
            Ptmp += self._probs[i] * (Pi + np.outer(err, err))
        Ptmp = (Ptmp + Ptmp.T) / 2
        self._post_cov = Ptmp

    def __innov_update(self):
        innov_org = [self._models[i].innov for i in range(self._models_n)]
        innov_cov_org = [self._models[i].innov_cov for i in range(self._models_n)]

        itmp = 0
        for i in range(self._models_n):
            itmp += self._probs[i] * innov_org[i]
        self._innov = itmp

        ictmp = 0
        for i in range(self._models_n):
            err = innov_org[i] - itmp
            ictmp += self._probs[i] * (innov_cov_org[i] + np.outer(err, err))
        ictmp = (ictmp + ictmp.T) / 2
        self._innov_cov = ictmp

    def _set_post_state(self, state):
        if self._models_n == 0:
            raise AttributeError("AttributeError: can't set attribute")
        for i in range(self._models_n):
            xi = self._switch_fcn(state, self._model_types[0], self._model_types[i])
            self._models[i].post_state = xi
    
    def _set_post_cov(self, cov):
        if self._models_n == 0:
            raise AttributeError("AttributeError: can't set attribute")
        for i in range(self._models_n):
            Pi = self._switch_fcn(cov, self._model_types[0], self._model_types[i])
            self._models[i].post_cov = Pi
    
    # the innovation and its covariance of model with maximum model probability
    # def _get_innov(self):
    #     return self._models[np.argmax(self._probs)].innov

    # def _get_innov_cov(self):
    #     return self._models[np.argmax(self._probs)].innov_cov

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
        self.__post_update()
        self._len = 0
        self._stage = 0
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
        self._xdim = models[0].xdim
        self._zdim = models[0].zdim
        if probs is None:
            self._probs = np.ones(self._models_n) / self._models_n
        else:
            self._probs = np.copy(probs)

    def predict(self, u=None, **kwargs):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        for i in range(self._models_n):
            self._models[i].predict(u, **kwargs)
        # update prior state and covariance
        self.__prior_update()

        self._stage = 1

    def update(self, z, **kwargs):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        # update probability
        pdf = np.zeros(self._models_n)
        for i in range(self._models_n):
            self._models[i].update(z, **kwargs)
            r = self._models[i].innov
            S = self._models[i].innov_cov
            # If there is a singular value, exp will be very small and all values in the pdf will be 0,
            # then total defined below will be 0 and an ZeroDivisionError will occur.
            pdf[i] = np.exp(-r @ lg.inv(S) @ r / 2) / np.sqrt(lg.det(2 * np.pi * S))
        # update innovation and associated covariance before updating the model probability
        self.__innov_update()
        # update model probability
        self._probs *= pdf
        self._probs /= np.sum(self._probs)
        # update posterior state and covariance
        self.__post_update()

        self._len += 1
        self._stage = 0

    def step(self, z, u=None, **kwargs):
        assert (self._stage == 0)

        self.predict(u, **kwargs)
        self.update(z, **kwargs)

    def weighted_state(self):
        return self._post_state

    def models(self):
        return self._models

    def probs(self):
        return self._probs
