# -*- coding: utf-8 -*-
'''
The multiple model filter can use other types of Kalman filters as its submodels
for filtering. Currently supported filters are stardard Kalman filter(KFilter),
extended Kalman filter and unscented Kalman filter. For the non-linear system with
additive Gaussian noise, this multiple model filter can be used as Gaussian sum
filter by setting different initial state and convariance of each non-linear filters
or submodels.
'''
from __future__ import division, absolute_import, print_function


__all__ = ['MMFilter']

import numpy as np
import scipy.linalg as lg
from .base import KFBase
from .kf import KFilter
from .ekf import EKFilterAN, EKFilterNAN
from .ukf import UKFilterAN, UKFilterNAN


class MMFilter(KFBase):
    '''
    Hybrid multiple model filter
    '''
    def __init__(self):
        super().__init__()
        self._model = {}
        self._model_n = 0

    def __str__(self):
        msg = 'Hybrid multiple model filter:\n{\n  '
        if self._model_n < 10:
            sub = ['{}: model: {}, probability: {}'.format(i, self._model[i][0], self._model[i][1]) for i in range(self._model_n)]
            sub = '\n  '.join(sub)
            # [i for i in range(self._model_n)]
        else:
            sub = ['{}: model: {}, probability: {}'.format(i, self._model[i][0], self._model[i][1]) for i in range(3)]
            sub.append('...')
            sub.extend(['{}: model: {}, probability: {}'.format(i, self._model[i][0], self._model[i][1]) for i in range(self._model_n - 3, self._model_n)])
            sub = '\n  '.join(sub)
        msg += sub
        msg += '\n}'
        return msg

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self._model.values())

    def __getitem__(self, n):
        return self._model[n]

    def init(self, state=None, cov=None):
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
        if isinstance(state, list):
            pass
        elif isinstance(state, np.ndarray):
            state = [state] * self._model_n
        else:
            raise ValueError('state must be a ndarray, list, not %s' % state.__class__.__name__)
        if isinstance(cov, list):
            pass
        elif isinstance(cov, np.ndarray):
            cov = [cov] * self._model_n
        else:
            raise ValueError('cov must be a ndarray, list, not %s' % cov.__class__.__name__)

        for i in range(self._model_n):
            self._model[i][0].init(state[i], cov[i])
        self._len = 0
        self._stage = 0
        self._init = True

    def add_model(self, model, prob):
        '''
        Add new model

        Parameters
        ----------
        model : KFilter, EKFilterAN, EKFilterNAN, UKFilterAN, UKFilterNAN
            standard Kalman filter, extended Kalman filter or unscented Kalman filter
        probability : float
            model prior probability

        Returns
        -------
            None
        '''
        if isinstance(model, (KFilter, EKFilterAN, EKFilterNAN, UKFilterAN, UKFilterNAN)):
            self._model[self._model_n] = [model, prob]
            self._model_n += 1

    def predict(self, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        for i in range(self._model_n):
            self._model[i][0].predict(u, **kw)

        self._stage = 1

    def update(self, z, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        pdf = []
        for i in range(self._model_n):
            self._model[i][0].update(z, **kw)
            r = self._model[i][0].innov
            S = self._model[i][0].innov_cov
            # If there is a wild value, exp will be very small and all values in the pdf will be 0,
            # then total defined below will be 0 and an ZeroDivisionError will occur.
            pdf.append((np.exp(-r @ lg.inv(S) @ r / 2) / np.sqrt(lg.det(2 * np.pi * S))).item())

        # total probability
        total = 0
        for i in range(self._model_n):
            total += pdf[i] * self._model[i][1]
        # update all models' posterior probability
        for i in range(self._model_n):
            self._model[i][1] = pdf[i] * self._model[i][1] / total 

        self._len += 1
        self._stage = 0

    def step(self, z, u=None, **kw):
        assert (self._stage == 0)

        self.predict(u, **kw)
        self.update(z, **kw)
        
    @property
    def weighted_state(self):
        # weighted state estimate
        state = 0
        for i in range(self._model_n):
            state += self._model[i][1] * self._model[i][0].post_state
        return state

    @property
    def maxprob_state(self):
        # state estimate of models with maximum probability
        max_index = np.argmax([self._model[i][1] for i in range(self._model_n)])
        return self._model[max_index][0].post_state

    @property
    def prob(self):
        return [self._model[i][1] for i in range(self._model_n)]

    @property
    def prior_state(self):
        return [m.prior_state for m in self._model]

    @property
    def post_state(self):
        return [m.post_state for m in self._model]

    @property
    def prior_cov(self):
        return [m.prior_cov for m in self._model]

    @property
    def post_cov(self):
        return [m.post_cov for m in self._model]

    @property
    def innov(self):
        return [m.innov for m in self._model]

    @property
    def innov_cov(self):
        return [m.innov_cov for m in self._model]

    @property
    def gain(self):
        return [m.gain for m in self._model]
