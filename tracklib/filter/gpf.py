# -*- coding: utf-8 -*-
'''
Gaussian particle filter
In fact, Gaussian particle filter belongs to Kalman filter rather than particle filter,
because Monte Carlo method is only used to propagate expectation and convariance, then
posterior probability density is approximated by Gaussian with the expectation and covariance
rather than samples(or particle) and weights.

REFERENCE:
[1]. J. H. Kotecha and P. M. Djuric, "Gaussian particle filtering," in IEEE Transactions on Signal Processing, vol. 51, no. 10, pp. 2592-2601, Oct. 2003.
'''
from __future__ import division, absolute_import, print_function


__all__ = ['GPFilter']

import numpy as np
import scipy.linalg as lg
from .base import FilterBase
from tracklib.utils import multi_normal, disc_random


class GPFilter(FilterBase):
    '''
    Gaussian particle filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1) + L_k-1*w_k-1
    z_k = h_k(x_k) + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k and v_k are additive noise
    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, f, L, h, M, Q, R, Ns):
        super().__init__()

        self._f = f
        self._L = L.copy()
        self._h = h
        self._M = M.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._Ns = Ns

    def __str__(self):
        msg = 'Gaussian particle filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        self._state = state.copy()
        self._cov = cov.copy()
        self._samples = np.empty((self._Ns, len(state)))
        self._weights = np.empty(self._Ns)
        self._init = True

    def reset(self, state, cov):
        self._state = state.copy()
        self._cov = cov.copy()

    def predict(self, u=None, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'L' in kwargs: self._L[:] = kwargs['L']
            if 'Q' in kwargs: self._Q[:] = kwargs['Q']

        Q_tilde = self._L @ self._Q @ self._L.T
        # draw samples from the posterior density
        self._samples[:] = multi_normal(self._state, self._cov, self._Ns, axis=0)
        # draw samples from prior density by drawing samples from transition density conditioned on
        # posterior samples drawn above. And the prior samples can be used in update step.
        proc_noi = multi_normal(0, Q_tilde, self._Ns, axis=0)
        for i in range(self._Ns):
            self._samples[i] = self._f(self._samples[i], u) + proc_noi[i]

        # compute prior_state and prior_cov, for coasted
        self._state = np.sum(self._samples, axis=0) / self._Ns
        self._cov = 0
        for i in range(self._Ns):
            err = self._samples[i] - self._state
            self._cov += np.outer(err, err)
        self._cov /= self._Ns
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov

    def correct(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'M' in kwargs: self._M[:] = kwargs['M']
            if 'R' in kwargs: self._R[:] = kwargs['R']

        # update weights to approximate the posterior density
        R_tilde = self._M @ self._R @ self._M.T
        for i in range(self._Ns):
            # this is not the innovation, just likelihood of measurement for each predicted sample
            noi = z - self._h(self._samples[i])
            pdf = 1 / np.sqrt(lg.det(2 * np.pi * R_tilde))
            pdf *= np.exp(-noi @ lg.inv(R_tilde) @ noi / 2)
            self._weights[i] = max(pdf, np.finfo(pdf).min)
        self._weights /= np.sum(self._weights)    # normalize

        # compute post_state and post_cov and the samples have been drawn in predict step
        self._state = np.dot(self._weights, self._samples)
        self._cov = 0
        for i in range(self._Ns):
            err = self._samples[i] - self._state
            self._cov += self._weights[i] * np.outer(err, err)
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov

    def distance(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        M = kwargs['M'] if 'M' in kwargs else self._M
        R = kwargs['R'] if 'R' in kwargs else self._R

        R_tilde = M @ R @ M.T
        h_map = [self._h(self._samples[i]) for i in range(self._Ns)]
        z_pred = np.sum(h_map, axis=0) / self._Ns
        # this is the real innovation, and the covariance fo innovation is uitilized to calculate
        # the likelihood of measurement for predicted measurement
        innov = z - z_pred
        S = 0
        for i in range(self._Ns):
            err = h_map[i] - z_pred
            S += np.outer(err, err)
        S = S / self._Ns + R_tilde
        S = (S + S.T) / 2
        d = innov @ lg.inv(S) @ innov + np.log(lg.det(S))

        return d

    def likelihood(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        M = kwargs['M'] if 'M' in kwargs else self._M
        R = kwargs['R'] if 'R' in kwargs else self._R

        R_tilde = M @ R @ M.T
        h_map = [self._h(self._samples[i]) for i in range(self._Ns)]
        z_pred = np.sum(h_map, axis=0) / self._Ns
        innov = z - z_pred
        S = 0
        for i in range(self._Ns):
            err = h_map[i] - z_pred
            S += np.outer(err, err)
        S = S / self._Ns + R_tilde
        S = (S + S.T) / 2
        pdf = 1 / np.sqrt(lg.det(2 * np.pi * S))
        pdf *= np.exp(-innov @ lg.inv(S) @ innov / 2)

        return max(pdf, np.finfo(pdf).min)     # prevent likelihood from being too small
