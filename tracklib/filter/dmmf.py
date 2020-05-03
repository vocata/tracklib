# -*- coding: utf-8 -*-
'''
Dynamic multiple model filter

REFERENCE:
[1]. Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation: Theory, Algorithms and Software," New York: Wiley, 2001
'''
from __future__ import division, absolute_import, print_function


__all__ = ['IMMFilter']

import copy
import numpy as np
import scipy.linalg as lg
import tracklib.model as model
from .base import KFBase


class IMMFilter(KFBase):
    '''
    Interacting multiple model filter
    '''
    def __init__(self, switch_fcn=model.model_switch):
        super().__init__()
        self._switch_fcn = switch_fcn

        self._models = []
        self._model_types = []
        self._probs = None
        self._trans_mat = None
        self._models_n = 0

    def __str__(self):
        msg = 'Interacting multiple model filter:\n{\n  '
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
            pi = self._switch_fcn(cov_org[i], types[i], types[0])
            err = xi - self._prior_state
            Ptmp += self._probs[i] * (pi + np.outer(err, err))
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
            pi = self._switch_fcn(cov_org[i], types[i], types[0])
            err = xi - self._post_state
            Ptmp += self._probs[i] * (pi + np.outer(err, err))
        Ptmp = (Ptmp + Ptmp.T) / 2
        self._post_cov = Ptmp

    def __innov_update(self):
        self._innov = 0
        itmp = 0
        for i in range(self._models_n):
            itmp += self._probs[i] * self._models[i].innov
        self._innov = itmp

        ictmp = 0
        for i in range(self._models_n):
            err = self._models[i].innov - self._innov
            ictmp += self._probs[i] * (self._models[i].innov_cov + np.outer(err, err))
        ictmp = (ictmp + ictmp.T) / 2
        self._innov_cov = ictmp

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
            raise RuntimeError('models must be added before calling init')

        for i in range(self._models_n):
            x = self._switch_fcn(state, self._model_types[0], self._model_types[i])
            P = self._switch_fcn(cov, self._model_types[0], self._model_types[i])
            self._models[i].init(x, P)
        self._post_state = state.copy()
        self._post_cov = cov.copy()
        self._len = 0
        self._stage = 0
        self._init = True

    def add_models(self, models, model_types, probs=None, trans_mat=None):
        '''
        Add new model

        Parameters
        ----------
        models : list, of length N
            the list of Kalman filter
        model_types : list, of length N
            the types corresponding to models
        probs : 1-D array_like, of length N, optional
            model probability
        trans_mat : 2-D array_like, of shape (N, N), optional
            model transition matrix

        Returns
        -------
            None
        '''
        self._models_n = len(models)
        self._models.extend(models)
        self._model_types.extend(model_types)
        self._xdim = models[0].xdim
        if probs is None:
            self._probs = np.ones(self._models_n) / self._models_n
        else:
            self._probs = np.copy(probs)
        if trans_mat is None:
            trans_prob = 0.999
            self._trans_mat = np.zeros((self._models_n, self._models_n))
            self._trans_mat += (1 - trans_prob) / 2
            idx = np.arange(self._models_n)
            self._trans_mat[idx, idx] = trans_prob
        else:
            self._trans_mat = np.copy(trans_mat)

    def predict(self, u=None):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        # mixing/interaction, the difference from the GPB1 and GPB2 is that merging
        # process (called mixing here) is carried out at the beginning of cycle.
        mixing_probs = self._trans_mat * self._probs
        # prior model probability P(M(k)|Z^(k-1))
        self._probs = np.sum(mixing_probs, axis=1)
        # mixing probability P(M(k-1)|M(k),Z^(k-1))
        mixing_probs /= self._probs.reshape(-1, 1)
        # mixing
        state_org = [self._models[i].post_state for i in range(self._models_n)]
        cov_org = [self._models[i].post_cov for i in range(self._models_n)]
        types = [self._model_types[i] for i in range(self._models_n)]
        for i in range(self._models_n):
            xi = 0
            for j in range(self._models_n):
                xj = self._switch_fcn(state_org[j], types[j], types[i])
                xi += mixing_probs[i, j] * xj
            self._models[i].post_state = xi
        for i in range(self._models_n):
            Pi = 0
            for j in range(self._models_n):
                xj = self._switch_fcn(state_org[j], types[j], types[i])
                Pj = self._switch_fcn(cov_org[j], types[j], types[i])
                err = xj - self._models[i].post_state
                Pi += mixing_probs[i, j] * (Pj + np.outer(err, err))
            Pi = (Pi + Pi.T) / 2
            self._models[i].post_cov = Pi

        for i in range(self._models_n):
            self._models[i].predict(u)
        self.__prior_update()

        self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        pdf = np.zeros(self._models_n)
        for i in range(self._models_n):
            self._models[i].update(z)
            r = self._models[i].innov
            S = self._models[i].innov_cov
            pdf[i] = np.exp(-r @ lg.inv(S) @ r / 2) / np.sqrt(lg.det(2 * np.pi * S))
        self.__innov_update()
        # posterior model probability P(M(k)|Z^k)
        self._probs *= pdf
        self._probs /= np.sum(self._probs)
        self.__post_update()

        self._len += 1
        self._stage = 0

    def step(self, z, u=None):
        assert (self._stage == 0)

        self.predict(u)
        self.update(z)

    def models(self):
        return self._models

    def probs(self):
        return self._probs

    def trans_mat(self):
        return self._trans_mat
