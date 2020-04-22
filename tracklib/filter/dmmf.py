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
from .base import KFBase


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
            raise RuntimeError('models must be added before calling init')

        for i in range(self._models_n):
            self._models[i].init(state, cov)
        self._post_state = state
        self._post_cov = cov
        self._len = 0
        self._stage = 0
        self._init = True

    def add_models(self, models, probs, trans_mat):
        '''
        Add new model

        Parameters
        ----------
        models : list, of length N
            the list of Kalman filter
        probs : list
            model prior probability
        trans_mat : 2-D array_like, of shape (N, N)
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
        if not isinstance(trans_mat, np.ndarray):
            raise TypeError('trans_mat must be a ndarray, not %s' %
                            trans_mat.__class__.__name__)
        if len(models) != len(probs):
            raise ValueError('the length of models must be the same as probs')

        self._models_n = len(models)
        self._models.extend(models)
        self._probs.extend(probs)
        self._trans_mat = trans_mat

    # single model is time-invariant, so kw is not required
    def predict(self, u=None):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        for i in range(self._models_n):
            self._models[i].predict(u)

        self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        # update probability
        pdf = np.zeros(self._models_n)
        prior_porb = np.dot(self._trans_mat, self._probs)
        for i in range(self._models_n):
            self._models[i].update(z)
            r = self._models[i].innov
            S = self._models[i].innov_cov
            pdf[i] = np.exp(-r @ lg.inv(S) @ r / 2) / np.sqrt(lg.det(2 * np.pi * S))
            self._probs[i] = pdf[i] * prior_porb[i]
        # normalize
        self._probs[:] /= np.sum(self._probs)

        # weighted posterior state and covariance
        self._post_state = 0
        for i in range(self._models_n):
            self._post_state += self._probs[i] * self._models[i].post_state

        # unlike before, posterior covariance is not equivalent to error covariance
        self._post_cov = 0
        for i in range(self._models_n):
            err = self._models[i].post_state - self._post_state
            self._post_cov += self._probs[i] * (self._models[i].post_cov + np.outer(err, err))
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

        # reset all models' posterior state and covariance
        for i in range(self._models_n):
            self._models[i].post_state = self._post_state
            self._models[i].post_cov = self._post_cov

        self._len += 1
        self._stage = 0

    def step(self, z, u=None):
        assert (self._stage == 0)

        self.predict(u)
        self.update(z)

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
    def trans_mat(self):
        if self._trans_mat is None:
            raise AttributeError("'%s' object has no attribute 'trans_mat'" %
                                 self.__class__.__name__)
        return self._trans_mat


class GPB2Filter(KFBase):
    '''
    Second-order generalized pseudo-Bayesian filter
    '''
    def __init__(self):
        super().__init__()
        self._models = []
        self._probs = []
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
                self._models[i][j].init(state, cov)
        self._post_state = state
        self._post_cov = cov
        self._len = 0
        self._stage = 0
        self._init = True

    def add_models(self, models, probs, trans_mat):
        '''
        Add new model

        Parameters
        ----------
        models : list, of length N
            the list of Kalman filter
        probs : list
            model prior probability
        trans_mat : 2-D array_like, of shape (N, N)
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
        if not isinstance(trans_mat, np.ndarray):
            raise TypeError('trans_mat must be a ndarray, not %s' %
                            trans_mat.__class__.__name__)
        if len(models) != len(probs):
            raise ValueError('the length of models must be the same as probs')

        self._models_n = len(models)
        for i in range(self._models_n):
            self._models.append([copy.deepcopy(models[i]) for _ in range(self._models_n)])
        self._probs.extend(probs)
        self._trans_mat = trans_mat

    def predict(self, u=None):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        for i in range(self._models_n):
            for j in range(self._models_n):
                self._models[i][j].predict(u)

        self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        # update probability
        pdf = np.zeros((self._models_n, self._models_n))
        merge_prob = np.zeros((self._models_n, self._models_n))
        for i in range(self._models_n):
            for j in range(self._models_n):
                self._models[i][j].update(z)
                r = self._models[i][j].innov
                S = self._models[i][j].innov_cov
                pdf[i][j] = np.exp(-r @ lg.inv(S) @ r / 2) / np.sqrt(lg.det(2 * np.pi * S))
                merge_prob[i][j] = pdf[i][j] * self._trans_mat[i][j] * self._probs[j]
        # normalize
        self._probs[:] = np.sum(merge_prob, axis=1)
        merge_prob /= np.reshape(self._probs, (-1, 1))
        self._probs /= np.sum(self._probs)

        # merge post state and covariance
        x_dim = len(self._post_state)
        merge_state = np.zeros((x_dim, self._models_n))
        for i in range(self._models_n):
            for j in range(self._models_n):
                merge_state[:, i] += merge_prob[i, j] * self._models[i][j].post_state
        self._post_state = np.dot(merge_state, self._probs)

        self._post_cov = 0
        merge_cov = np.zeros((x_dim, x_dim, self._models_n))
        for i in range(self._models_n):
            for j in range(self._models_n):
                errj = self._models[i][j].post_state - merge_state[:, i]
                merge_cov[:, :, i] += merge_prob[i, j] * (self._models[i][j].post_cov + np.outer(errj, errj))
            merge_cov[:, :, i] = (merge_cov[:, :, i] + merge_cov[:, :, i].T) / 2
            erri = merge_state[:, i] - self._post_state
            self._post_cov += self._probs[i] * (merge_cov[:, :, i] + np.outer(erri, erri))
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

        # reset all models' posterior state and covariance
        for i in range(self._models_n):
            for j in range(self._models_n):
                self._models[i][j].post_state = merge_state[:, j]
                self._models[i][j].post_cov = merge_cov[:, :, j]

        self._len += 1
        self._stage = 0

    def step(self, z, u=None):
        assert(self._stage == 0)

        self.predict(u)
        self.update(z)

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
    def trans_mat(self):
        if self._trans_mat is None:
            raise AttributeError("'%s' object has no attribute 'trans_mat'" %
                                 self.__class__.__name__)
        return self._trans_mat


class IMMFilter(KFBase):
    pass