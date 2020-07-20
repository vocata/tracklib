# -*- coding: utf-8 -*-
'''
Extended object tracker

REFERENCE:
[1]. 
'''
from __future__ import division, absolute_import, print_function


__all__ = ['EOPFilter', 'IMMEOPFilter']

import numbers
import numpy as np
import scipy.linalg as lg
import scipy.stats as st
from .base import EOFilterBase
from tracklib.utils import disc_random, ellipsoidal_volume


class EOPFilter(EOFilterBase):
    '''
    Extended object particle filter
    '''
    def __init__(self, F, H, Q, R, Ns, Neff, df, lamb=None, resample_alg='roulette'):
        self._F = F.copy()
        self._H = H.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._Ns = Ns
        self._Neff = Neff
        self._df = df
        self._lamb = lamb
        self._resample_alg = resample_alg
        self._init = False

    def init(self, state, cov, df, extension):
        self._state = state.copy()
        self._cov = cov.copy()
        self._ext = extension.copy()

        self._state_samples = st.multivariate_normal.rvs(state, cov, self._Ns)
        self._ext_samples = st.wishart.rvs(df, extension / df, self._Ns)
        self._weights = np.full(self._Ns, 1 / self._Ns, dtype=float)
        self._init = True

    def predict(self):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        # update samples
        self._ext_samples[:] = [
            st.wishart.rvs(self._df, self._ext_samples[i] / self._df)
            for i in range(self._Ns)
        ]
        self._state_samples[:] = [
            st.multivariate_normal.rvs(np.dot(self._F, self._state_samples[i]),
                                       np.kron(self._ext_samples[i], self._Q))
            for i in range(self._Ns)
        ]

        # compute prior extension, state and covariance
        self._ext = 0
        self._state = 0
        for i in range(self._Ns):
            self._ext += self._weights[i] * self._ext_samples[i]
            self._state += self._weights[i] * self._state_samples[i]
        self._cov = 0
        for i in range(self._Ns):
            err = self._state_samples[i] - self._state
            self._cov += self._weights[i] * np.outer(err, err)
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov, self._ext

    def correct(self, zs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        Nm = len(zs)    # measurements number
        if self._lamb is None:
            lamb = Nm / ellipsoidal_volume(self._ext)        # empirical target density
        else:
            lamb = self._lamb

        # update weights
        const_arr = np.zeros(self._Ns)
        dist_arr = np.zeros(self._Ns)
        for i in range(self._Ns):
            cov = self._ext_samples[i] / 4 + self._R
            cov_inv = lg.inv(cov)
            V = ellipsoidal_volume(self._ext_samples[i])
            pmf = st.poisson.pmf(Nm, lamb * V)
            # pmf = 1
            const_arr[i] = pmf / lg.det(2 * np.pi * cov)**(Nm / 2)

            dist = 0
            for j in range(Nm):
                d = zs[j] - np.dot(self._H, self._state_samples[i])
                dist += d @ cov_inv @ d / 2
            dist_arr[i] = dist
        # the underflow problem is avoided by adding offset to exp function
        exp_term = np.exp(-dist_arr + dist_arr.min())
        self._weights *= const_arr * exp_term
        self._weights /= self._weights.sum()

        # resample
        Neff = 1 / (self._weights**2).sum()
        if Neff <= self._Neff:
            self._state_samples[:], index = disc_random(self._weights,
                                                        self._Ns,
                                                        self._state_samples,
                                                        alg=self._resample_alg)
            self._ext_samples[:] = self._ext_samples[index]
            self._weights[:] = 1 / self._Ns

        # compute posterior extension, state and covariance
        self._ext = 0
        self._state = 0
        for i in range(self._Ns):
            self._ext += self._weights[i] * self._ext_samples[i]
            self._state += self._weights[i] * self._state_samples[i]
        self._cov = 0
        for i in range(self._Ns):
            err = self._state_samples[i] - self._state
            self._cov += self._weights[i] * np.outer(err, err)
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov, self._ext

    def distance(self, z, **kwargs):
        return super().distance(z, **kwargs)

    def likelihood(self, z, **kwargs):
        return super().likelihood(z, **kwargs)


class IMMEOPFilter(EOFilterBase):
    '''
    Extended object particle filter
    '''
    def __init__(self,
                 models_n,
                 init_fcn,
                 state_trans_fcn,
                 ext_trans_fcn,
                 meas_fcn,
                 merge_fcn,
                 df,
                 state_noise,
                 meas_noise,
                 Ns,
                 Neff,
                 lamb=None,
                 trans_mat=0.9,
                 probs=None,
                 resample_alg='roulette'):
        super().__init__()

        self._models_n = models_n
        self._init_fcn = init_fcn
        self._state_trans_fcn = state_trans_fcn
        self._ext_trans_fcn = ext_trans_fcn
        self._meas_fcn = meas_fcn
        self._merge_fcn = merge_fcn
        self._df = df
        self._state_noise = state_noise
        self._meas_noise = meas_noise
        self._Ns = Ns
        self._Neff = Neff
        self._lamb = lamb
        if models_n == 1:
            self._trans_mat = np.eye(1)
        elif isinstance(trans_mat, numbers.Number):
            other_probs = (1 - trans_mat) / (models_n - 1)
            self._trans_mat = np.full((models_n, models_n), other_probs, dtype=float)
            np.fill_diagonal(self._trans_mat, trans_mat)
        else:
            self._trans_mat = trans_mat
        if probs is None:
            self._probs = np.full(models_n, 1 / models_n, dtype=float)
        else:
            self._probs = probs
        self._resample_alg = resample_alg

        self._init = False

    def init(self, state, cov, df, extension):
        self._state = state.copy()
        self._cov = cov.copy()
        self._ext = extension.copy()

        self._index = np.zeros(self._Ns, dtype=int)
        self._index[:], _ = disc_random(self._probs, self._Ns, alg='low_var')

        self._state_samples, self._ext_samples = self._init_fcn(state, cov, df, extension, self._Ns)
        self._weights = np.full(self._Ns, 1 / self._Ns, dtype=float)

        self._init = True

    def predict(self):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        # update samples
        index_bak = self._index.copy()
        for i in range(self._Ns):
            idx, _ = disc_random(self._trans_mat[:, self._index[i]])
            self._index[i] = idx[0]
        for i in range(self._Ns):
            idx = self._index[i]
            self._ext_samples[i] = self._ext_trans_fcn[idx](
                self._ext_samples[i], self._state_samples[i],
                self._df[idx], index_bak[i])
        for i in range(self._Ns):
            idx = self._index[i]
            self._state_samples[i] = self._state_trans_fcn[idx](
                self._state_samples[i], self._ext_samples[i],
                self._state_noise[idx], index_bak[i])

        # compute prior model probability
        for i in range(self._models_n):
            self._probs[i] = (self._index == i).sum()
        self._probs /= self._probs.sum()

        # compute prior extension, state and covariance
        self._state, self._cov, self._ext = self._merge_fcn(
            self._state_samples, self._ext_samples, self._weights, self._index,
            self._Ns)

        return self._state, self._cov, self._ext

    def correct(self, zs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        # update weights
        Nm = len(zs)    # measurements number
        if self._lamb is None:
            lamb = Nm / ellipsoidal_volume(self._ext)        # empirical target density
        else:
            lamb = self._lamb

        # update weights
        const_arr = np.zeros(self._Ns)
        dist_arr = np.zeros(self._Ns)
        for i in range(self._Ns):
            cov = self._ext_samples[i] / 4 + self._meas_noise[self._index[i]]
            cov_inv = lg.inv(cov)
            V = ellipsoidal_volume(self._ext_samples[i])
            pmf = st.poisson.pmf(Nm, lamb * V)
            # pmf = 1
            const_arr[i] = pmf / lg.det(2 * np.pi * cov)**(Nm / 2)

            dist = 0
            for j in range(Nm):
                d = zs[j] - self._meas_fcn[self._index[i]](self._state_samples[i])
                dist += d @ cov_inv @ d / 2
            dist_arr[i] = dist
        # the underflow problem is avoided by adding offset to exp function
        exp_term = np.exp(-dist_arr + dist_arr.min())
        self._weights *= const_arr * exp_term
        self._weights /= self._weights.sum()

        # resample
        Neff = 1 / (self._weights**2).sum()
        if Neff <= self._Neff:
            self._state_samples[:], index = disc_random(self._weights,
                                                        self._Ns,
                                                        self._state_samples,
                                                        alg=self._resample_alg)
            self._ext_samples[:] = self._ext_samples[index]
            self._index[:] = self._index[index]
            self._weights[:] = 1 / self._Ns

        # compute posterior model probability
        for i in range(self._models_n):
            self._probs[i] = (self._index == i).sum()
        self._probs /= self._probs.sum()

        # compute posterior extension, state and covariance
        self._state, self._cov, self._ext = self._merge_fcn(
            self._state_samples, self._ext_samples, self._weights, self._index,
            self._Ns)

        return self._state, self._cov, self._ext

    def distance(self, z, **kwargs):
        return super().distance(z, **kwargs)

    def likelihood(self, z, **kwargs):
        return super().likelihood(z, **kwargs)

    def probs(self):
        return self._probs

    def trans_mat(self):
        return self._trans_mat