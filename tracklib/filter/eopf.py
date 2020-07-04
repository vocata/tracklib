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
from tracklib.utils import disc_random, ellipsoidal_volume


class EOPFilter(FilterBase):
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

    def init(self, state, cov, extension):
        self._state = state.copy()
        self._cov = cov.copy()
        self._ext = extension.copy()
        self._post_ext = extension.copy()

        self._state_samples = st.multivariate_normal.rvs(state, cov, self._Ns)
        self._ext_samples = st.wishart.rvs(self._df, extension / self._df, self._Ns)
        self._weights = np.full(self._Ns, 1 / self._Ns, dtype=float)
        self._init = True

    def reset(self, state, cov, extension):
        self._state = state.copy()
        self._cov = cov.copy()
        self._ext = extension.copy()
        self._post_ext = extension.copy()

        self._state_samples = st.multivariate_normal.rvs(state, cov, self._Ns)
        self._ext_samples = st.wishart.rvs(self._df, extension / self._df, self._Ns)
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
        self._ext_samples[:] = [
            st.wishart.rvs(self._df, self._ext_samples[i] / self._df) +
            0.5 * np.eye(*self._ext_samples[i].shape) for i in range(self._Ns)          # prevent the ellipse from being too small
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

        # for i in range(self._Ns):
        #     B = st.wishart.rvs(self._df, np.eye(2) / self._df)
        #     self._ext_samples[i] = B @ self._ext_samples[i] @ B.T
        #     # self._ext_samples[i] = B @ self._ext_samples[i] @ B.T + 1 * np.eye(2)

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
            lamb = Nm / ellipsoidal_volume(self._post_ext)        # empirical target density
        else:
            lamb = self._lamb

        # update weights
        const_arr = np.zeros(self._Ns)
        dist_arr = np.zeros(self._Ns)
        for i in range(self._Ns):
            cov = self._ext_samples[i] + self._R
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

        self._post_ext = self._ext

        return self._state, self._cov, self._ext

    def distance(self, z, **kwargs):
        return super().distance(z, **kwargs)

    def likelihood(self, z, **kwargs):
        return super().likelihood(z, **kwargs)

    def extension(self):
        return self._ext