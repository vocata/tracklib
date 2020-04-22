'''
Dynamic multiple model filter including GPB1, GPB2 and IMM

REFERENCE:
[1]. Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation: Theory, Algorithms and Software," New York: Wiley, 2001
'''
from __future__ import division, absolute_import, print_function


__all__ = ['GPB1Filter', 'GPB2Filter', 'IMMFilter']

import numpy as np
import scipy.linalg as lg
from .base import KFBase
from .kf import KFilter
from .ekf import EKFilterAN, EKFilterNAN
from .ukf import UKFilterAN, UKFilterNAN


class GPB1Filter(KFBase):
    '''
    First-order generalized pseudo-Bayesian filter
    '''
    def __init__(self):
        super().__init__()
        self._models = []
        self._probs = []
        self._trans_mat = None
        self._models_n = 0

    def __str__(self):
        msg = 'First-order generalized pseudo-Bayesian filter:\n{\n  '
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
        state : ndarray
            Initial prior state estimate
        cov : ndarray
            Initial error convariance matrix

        Returns
        -------
            None
        '''
        if self._models_n == 0:
            raise RuntimeError('submodel must be added before calling init')

        for i in range(self._models_n):
            self._models[i].init(state, cov)
        self._post_state = state
        self._post_cov = cov
        self._len = 0
        self._stage = 0
        self._init = True

    def add_models(self, models, probs, transition_matrix):
        '''
        Add new model

        Parameters
        ----------
        models : list, of length N
            the list of Kalman filter
        probs : list
            model prior probability
        transition_matrix : 2-D array_like, of shape (N, N)
            model transition matrix

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
        if not isinstance(transition_matrix, np.ndarray):
            raise TypeError('transition_matrix must be a ndarray, not %s' %
                            transition_matrix.__class__.__name__)
        if len(models) != len(probs):
            raise ValueError('the length of models must be the same as probs')

        self._models.extend(models)
        self._probs.extend(probs)
        self._trans_mat = transition_matrix
        self._models_n = len(models)

    # single model is time-invariant, so kw is not required
    def predict(self, u=None):
        assert(self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        for i in range(self._models_n):
            self._models[i].predict(u)

        self._stage = 1

    def update(self, z):
        assert(self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        pdf = np.zeros(self._models_n)
        for i in range(self._models_n):
            self._models[i][0].update(z)
            r = self._models[0]._innov
            S = self._models[0]._innov_cov
            pdf[i] = np.exp(-r @ lg.inv(S) @ r / 2) / np.sqrt(lg.det(2 * np.pi * S))

        # total probability
        prior_porb = np.dot(self._trans_mat, self._probs)
        total = np.dot(pdf, prior_porb)
        # update all models' posterior probability
        for i in range(self._models_n):
            self._probs[i] = pdf[i] * prior_porb[i] / total

        # weighted posterior state and covariance
        self._post_state = 0
        for i in range(self._models_n):
            self._post_state += self._probs[i] * self._models[i]._post_state
        # Unlike before, posterior covariance is not equivalent to error covariance
        self._post_cov = 0
        for i in range(self._models_n):
            err = self._models[i]._post_state - self._post_state
            self._post_cov += self._probs[i] * (self._models[i]._post_cov + np.outer(err, err))

        # reset models posterior state and covariance
        for i in range(self._models_n):
            self._models[i]._post_state = self._post_state
            self._models[i]._post_cov = self._post_cov

    def step(self, z, u=None):
        assert (self._stage == 0)

        self.predict(u)
        self.update(z)

    @property
    def models(self):
        return self._models

    @property
    def probs(self):
        return self._probs
