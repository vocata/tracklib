# -*- coding: utf-8 -*-
'''
Extended object tracker

REFERENCE:
[1]. 
'''
from __future__ import division, absolute_import, print_function


__all__ = ['EOPFilter']

import numpy as np
import scipy.linalg as lg
import scipy.stats as st
import scipy.special as sl
from .base import FilterBase
from tracklib.utils import disc_random


class EOPFilter(FilterBase):
    '''
    Extended object particle filter
    '''
    def __init__(self, F, H, Q, R, Ns, Neff, df, lamb, resample_alg='roulette'):
        self._F = F.copy()
        self._H = H.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._Ns = Ns
        self._Neff = Neff
        self._df = df
        self._lamb = lamb
        self._resample_alg = resample_alg

    def init(self, state, cov, extension, df):
        self._state = state.copy()
        self._cov = cov.copy()
        self._ext = extension.copy()

        self._state_samples = st.multivariate_normal.rvs(state, cov, self._Ns)
        self._ext_samples = st.invwishart.rvs(df, extension * df, self._Ns)
        self._weights = np.full(self._Ns, 1 / self._Ns, dtype=float)
        self._init = True

    def reset(self, state, cov, extension, df):
        self._state = state.copy()
        self._cov = cov.copy()
        self._ext = extension.copy()

        self._state_samples = st.multivariate_normal.rvs(state, cov, self._Ns)
        self._ext_samples = st.invwishart.rvs(df, extension * df, self._Ns)
        self._weights = np.full(self._Ns, 1 / self._Ns, dtype=float)

    def predict(self):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        # update samples
        self._state_samples[:] = [
            st.multivariate_normal.rvs(np.dot(self._F, self._state_samples[i]),
                                       np.kron(self._ext_samples[i], self._Q))
            for i in range(self._Ns)
        ]
        # ext_samples = []
        # for i in range(self._Ns):
        #     try:
        #         # ext = st.wishart.rvs(self._df, self._ext_samples[i] / self._df)
        #         G = st.multivariate_normal.rvs(cov=self._ext_samples[i] / self._df, size=self._df)
        #         ext = np.dot(G.T, G)
        #         ext_samples.append(ext)
        #     except lg.LinAlgError:
        #         ext_samples.append(self._ext_samples[i])
        #         print('in')
        # self._ext_samples = np.array(ext_samples)
        # self._ext_samples[:] = [
        #     st.wishart.rvs(self._df, self._ext_samples[i] / self._df)
        #     for i in range(self._Ns)
        # ]

        for i in range(self._Ns):
            B = st.wishart.rvs(self._df, np.eye(2) / self._df)
            self._ext_samples[i] = B @ self._ext_samples[i] @ B.T + 10 * np.eye(2)

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

    def __volume(self, X):
        dim = X.shape[0]
        n = dim / 2
        vol = np.pi**(n) * lg.det(X) / sl.gamma(n + 1)
        return vol

    def correct(self, zs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        # update weights
        Nm = len(zs)    # measurements number
        for i in range(self._Ns):
            pdf = 1
            # V = self.__volume(self._ext_samples[i])
            for j in range(Nm):
                pdf *= st.multivariate_normal.pdf(
                    zs[j], np.dot(self._H, self._state_samples[i]),
                    self._ext_samples[i] + self._R)
            # pdf *= st.poisson.pmf(Nm, self._lamb * V)     # for large extension or group
            self._weights[i] *= max(pdf, np.finfo(pdf).tiny)
        self._weights /= np.sum(self._weights)

        # resample
        Neff = 1 / np.sum(self._weights**2)
        if Neff <= self._Neff:
            # new_samples, _ = disc_random(self._weights,
            #                              self._Ns,
            #                              list(
            #                                  zip(self._state_samples,
            #                                      self._ext_samples)),
            #                              alg=self._resample_alg)
            # for i in range(self._Ns):
            #     self._state_samples[i] = new_samples[i][0]
            #     self._ext_samples[i] = new_samples[i][1]

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

    @property
    def extension(self):
        return self._ext