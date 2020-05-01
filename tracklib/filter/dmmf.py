# -*- coding: utf-8 -*-
'''
Dynamic multiple model filter including GPB1, GPB2 and IMM

REFERENCE:
[1]. Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation: Theory, Algorithms and Software," New York: Wiley, 2001
'''
from __future__ import division, absolute_import, print_function


__all__ = ['GPB1Filter', 'GPB2Filter', 'IMMFilter']

import copy
import numpy as np
import scipy.linalg as lg
import tracklib.model as model
from .base import KFBase


class GPB1Filter(KFBase):
    '''
    First-order generalized pseudo-Bayesian filter
    '''
    def __init__(self, xdim, zdim, switch_fcn=model.model_switch):
        super().__init__()
        self._xdim = xdim
        self._zdim = zdim
        self._switch_fcn = switch_fcn

        order = xdim // zdim
        if order == 1:
            self._state_type = 'cp'
        elif order == 2:
            self._state_type = 'cv'
        elif order == 3:
            self._state_type = 'ca'
        else:
            raise ValueError('xdim does not match zdim')
        self._models = []
        self._model_types = []
        self._probs = None
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

    def __prior_update(self):
        prior_state_cov = []
        for i in range(self._models_n):
            prior_state_cov.append(
                self._switch_fcn(self._models[i].prior_state,
                                 self._models[i].prior_cov,
                                 self._model_types[i], self._state_type))
        self._prior_state = 0
        for i in range(self._models_n):
            self._prior_state += self._probs[i] * prior_state_cov[i][0]
        self._prior_cov = 0
        for i in range(self._models_n):
            err = prior_state_cov[i][0] - self._prior_state
            self._prior_cov += self._probs[i] * (prior_state_cov[i][1] + np.outer(err, err))
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

    def __post_update(self):
        post_state_cov = []
        for i in range(self._models_n):
            post_state_cov.append(
                self._switch_fcn(self._models[i].post_state,
                                 self._models[i].post_cov,
                                 self._model_types[i], self._state_type))
        self._post_state = 0
        for i in range(self._models_n):
            self._post_state += self._probs[i] * post_state_cov[i][0]
        self._post_cov = 0
        for i in range(self._models_n):
            err = post_state_cov[i][0] - self._post_state
            self._post_cov += self._probs[i] * (post_state_cov[i][1] + np.outer(err, err))
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

    def __innov_update(self):
        self._innov = 0
        for i in range(self._models_n):
            self._innov += self._probs[i] * self._models[i].innov
        self._innov_cov = 0
        for i in range(self._models_n):
            err = self._models[i].innov - self._innov
            self._innov_cov += self._probs[i] * (self._models[i].innov_cov + np.outer(err, err))
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2

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
            self._models[i].init(*self._switch_fcn(
                state, cov, self._state_type, self._model_types[i]))
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
            the type of models
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

    # single model is time-invariant, so kw is not required
    def predict(self, u=None):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        for i in range(self._models_n):
            self._models[i].predict(u)
        # prior model probability P(M(k)|Z^(k-1))
        self._probs = np.dot(self._trans_mat, self._probs)
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

        # reset all models' posterior state and covariance
        for i in range(self._models_n):
            post_state, post_cov = self._switch_fcn(self._post_state,
                                                    self._post_cov,
                                                    self._state_type,
                                                    self._model_types[i])
            self._models[i].post_state = post_state
            self._models[i].post_cov = post_cov

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


class GPB2Filter(KFBase):
    '''
    Second-order generalized pseudo-Bayesian filter
    '''
    def __init__(self, xdim, zdim, switch_fcn=model.model_switch):
        super().__init__()
        self._xdim = xdim
        self._zdim = zdim
        self._switch_fcn = switch_fcn

        order = xdim // zdim
        if order == 1:
            self._state_type = 'cp'
        elif order == 2:
            self._state_type = 'cv'
        elif order == 3:
            self._state_type = 'ca'
        else:
            raise ValueError('xdim does not match zdim')
        self._models = []
        self._model_types = []
        self._probs = None
        self._prior_probs = None
        self._merge_probs = None
        self._prior_merge_probs = None
        self._trans_mat = None
        self._models_n = 0

    def __str__(self):
        msg = 'Second-order generalized pseudo-Bayesian filter:\n{\n  '
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
        prior_state_cov = []
        for i in range(self._models_n):
            tmp = []
            for j in range(self._models_n):
                tmp.append(
                    self._switch_fcn(self._models[i][j].prior_state,
                                     self._models[i][j].prior_cov,
                                     self._model_types[i], self._state_type))
            prior_state_cov.append(tmp)
        self._prior_state = 0
        merge_state = []
        for i in range(self._models_n):
            merge_state.append(0)
            for j in range(self._models_n):
                merge_state[i] += self._prior_merge_probs[i, j] * prior_state_cov[i][j][0]
            self._prior_state += self._prior_probs[i] * merge_state[i]

        self._prior_cov = 0
        for i in range(self._models_n):
            merge_cov = 0
            for j in range(self._models_n):
                errj = prior_state_cov[i][j][0] - merge_state[i]
                merge_cov += self._prior_merge_probs[i, j] * (prior_state_cov[i][j][1] + np.outer(errj, errj))
            merge_cov = (merge_cov + merge_cov.T) / 2
            erri = merge_state[i] - self._prior_state
            self._prior_cov += self._prior_probs[i] * (merge_cov + np.outer(erri, erri))
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

    def __post_update(self):
        post_state_cov = []
        for i in range(self._models_n):
            tmp = []
            for j in range(self._models_n):
                tmp.append(
                    self._switch_fcn(self._models[i][j].post_state,
                                     self._models[i][j].post_cov,
                                     self._model_types[i], self._state_type))
            post_state_cov.append(tmp)
        self._post_state = 0
        merge_state = []
        for i in range(self._models_n):
            merge_state.append(0)
            for j in range(self._models_n):
                merge_state[i] += self._merge_probs[i, j] * post_state_cov[i][j][0]
            self._post_state += self._probs[i] * merge_state[i]

        self._post_cov = 0
        merge_cov = []
        for i in range(self._models_n):
            merge_cov.append(0)
            for j in range(self._models_n):
                errj = post_state_cov[i][j][0] - merge_state[i]
                merge_cov[i] += self._merge_probs[i, j] * (post_state_cov[i][j][1] + np.outer(errj, errj))
            merge_cov[i] = (merge_cov[i] + merge_cov[i].T) / 2
            erri = merge_state[i] - self._post_state
            self._post_cov += self._probs[i] * (merge_cov[i] + np.outer(erri, erri))
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

        return merge_state, merge_cov

    def __innov_update(self):
        self._innov = 0
        merge_innov = []
        for i in range(self._models_n):
            merge_innov.append(0)
            for j in range(self._models_n):
                merge_innov[i] += self._prior_merge_probs[i, j] * self._models[i][j].innov
            self._innov += self._prior_probs[i] * merge_innov[i]

        self._innov_cov = 0
        for i in range(self._models_n):
            merge_cov = 0
            for j in range(self._models_n):
                errj = self._models[i][j].innov - merge_innov[i]
                merge_cov += self._prior_merge_probs[i, j] * (self._models[i][j].innov_cov + np.outer(errj, errj))
            merge_cov = (merge_cov + merge_cov.T) / 2
            erri = merge_innov[i] - self._innov
            self._innov_cov += self._prior_probs[i] * (merge_cov + np.outer(erri, erri))
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2

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
            for j in range(self._models_n):
                self._models[i][j].init(*self._switch_fcn(
                    state, cov, self._state_type, self._model_types[i]))
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
            the type of models
        probs : 1-D array_like, of length N, optional
            model probability
        trans_mat : 2-D array_like, of shape (N, N), optional
            model transition matrix

        Returns
        -------
            None
        '''
        self._models_n = len(models)
        for i in range(self._models_n):
            self._models.append([copy.deepcopy(models[i]) for _ in range(self._models_n)])
        self._model_types.extend(model_types)
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
        self._merge_probs = np.zeros((self._models_n, self._models_n))

    def predict(self, u=None):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        for i in range(self._models_n):
            for j in range(self._models_n):
                self._models[i][j].predict(u)
        # prior merge probability P(M(k-1)|M(k),Z^(k-1)) and prior model probability P(M(k)|Z^(k-1))
        self._prior_merge_probs = self._trans_mat * self._probs
        self._prior_probs = np.sum(self._prior_merge_probs, axis=1)
        self._prior_merge_probs /= self._prior_probs.reshape(-1, 1)
        self.__prior_update()

        self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        pdf = np.zeros((self._models_n, self._models_n))
        for i in range(self._models_n):
            for j in range(self._models_n):
                self._models[i][j].update(z)
                r = self._models[i][j].innov
                S = self._models[i][j].innov_cov
                pdf[i][j] = np.exp(-r @ lg.inv(S) @ r / 2) / np.sqrt(lg.det(2 * np.pi * S))
        self.__innov_update()
        # post merge probability P(M(k-1)|M(k),Z^k)) and post model probability P(M(k)|Z^k)
        self._merge_probs = pdf * self._trans_mat * self._probs
        self._probs = np.sum(self._merge_probs, axis=1)
        self._merge_probs /= self._probs.reshape(-1, 1)
        self._probs /= np.sum(self._probs)
        merge_state, merge_cov = self.__post_update()

        # reset all models' posterior state and covariance
        for i in range(self._models_n):
            post_state, post_cov = self._switch_fcn(merge_state[i],
                                                    merge_cov[i],
                                                    self._state_type,
                                                    self._model_types[i])
            for j in range(self._models_n):
                self._models[i][j].post_state = post_state
                self._models[i][j].post_cov = post_cov

        self._len += 1
        self._stage = 0

    def step(self, z, u=None):
        assert(self._stage == 0)

        self.predict(u)
        self.update(z)

    def models(self):
        return self._models

    def probs(self):
        return self._probs

    def trans_mat(self):
        return self._trans_mat


class IMMFilter(KFBase):
    '''
    Interacting multiple model filter
    '''
    def __init__(self, xdim, zdim, switch_fcn=model.model_switch):
        super().__init__()
        self._xdim = xdim
        self._zdim = zdim
        self._switch_fcn = switch_fcn

        order = xdim // zdim
        if order == 1:
            self._state_type = 'cp'
        elif order == 2:
            self._state_type = 'cv'
        elif order == 3:
            self._state_type = 'ca'
        else:
            raise ValueError('xdim does not match zdim')
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
        prior_state_cov = []
        for i in range(self._models_n):
            prior_state_cov.append(
                self._switch_fcn(self._models[i].prior_state,
                                 self._models[i].prior_cov,
                                 self._model_types[i], self._state_type))
        self._prior_state = 0
        for i in range(self._models_n):
            self._prior_state += self._probs[i] * prior_state_cov[i][0]
        self._prior_cov = 0
        for i in range(self._models_n):
            err = prior_state_cov[i][0] - self._prior_state
            self._prior_cov += self._probs[i] * (prior_state_cov[i][1] + np.outer(err, err))
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

    def __post_update(self):
        post_state_cov = []
        for i in range(self._models_n):
            post_state_cov.append(
                self._switch_fcn(self._models[i].post_state,
                                 self._models[i].post_cov,
                                 self._model_types[i], self._state_type))
        self._post_state = 0
        for i in range(self._models_n):
            self._post_state += self._probs[i] * post_state_cov[i][0]
        self._post_cov = 0
        for i in range(self._models_n):
            err = post_state_cov[i][0] - self._post_state
            self._post_cov += self._probs[i] * (post_state_cov[i][1] + np.outer(err, err))
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

    def __innov_update(self):
        self._innov = 0
        for i in range(self._models_n):
            self._innov += self._probs[i] * self._models[i].innov
        self._innov_cov = 0
        for i in range(self._models_n):
            err = self._models[i].innov - self._innov
            self._innov_cov += self._probs[i] * (self._models[i].innov_cov + np.outer(err, err))
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2

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
            self._models[i].init(*self._switch_fcn(
                state, cov, self._state_type, self._model_types[i]))
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
            the type of models
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
        post_state_cov = []
        for i in range(self._models_n):
            post_state_cov.append(
                self._switch_fcn(self._models[i].post_state,
                                 self._models[i].post_cov,
                                 self._model_types[i], self._state_type))
        mixed_state = []
        for i in range(self._models_n):
            mixed_state.append(0)
            for j in range(self._models_n):
                mixed_state[i] += mixing_probs[i, j] * post_state_cov[j][0]

        mixed_cov = []
        for i in range(self._models_n):
            mixed_cov.append(0)
            for j in range(self._models_n):
                err = post_state_cov[j][0] - mixed_state[i]
                mixed_cov[i] += mixing_probs[i, j] * (post_state_cov[j][1] + np.outer(err, err))
            mixed_cov[i] = (mixed_cov[i] + mixed_cov[i].T) / 2
            sub_state, sub_cov = self._switch_fcn(mixed_state[i], mixed_cov[i],
                                                  self._state_type,
                                                  self._model_types[i])
            self._models[i].post_state, self._models[i].post_cov = sub_state, sub_cov

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
