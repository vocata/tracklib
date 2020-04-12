# -*- coding: utf-8 -*-
'''
Gaussian particle filter
In fact, Gaussian particle filter belongs to Kalman filter rather than particle filter,
because Monte Carlo method is only used to propagate expectation and convariance, then
posterior probability density is approximated by Gaussian with the expectation and covariance
rather than samples(or particle) and weights.

REFERENCE:
[1]. Gaussian Particle Filtering
'''
from __future__ import division, absolute_import, print_function


__all__ = ['GPFilter']

import numpy as np
import scipy.linalg as lg
from .base import KFBase
from tracklib.utils import multi_normal, disc_random


class GPFilter(KFBase):
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
        self._L = L
        self._h = h
        self._M = M
        self._Q = Q
        self._R = R
        self._Ns = Ns

    def __str__(self):
        msg = 'Gaussian particle filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        self._prior_state = state
        self._prior_cov = cov
        self._post_state = state
        self._post_cov = cov
        self._samples = np.empty((self._Ns, len(state)))
        self._weights = np.empty(self._Ns)
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kw) > 0:
            if 'f' in kw: self._f = kw['f']
            if 'L' in kw: self._L = kw['L']
            if 'Q' in kw: self._Q = kw['Q']

        Q_tilde = self._L @ self._Q @ self._L.T
        # draw samples from the posterior density
        self._samples[:] = multi_normal(self._post_state, self._post_cov, self._Ns, axis=0)
        self._weights[:] = 1 / self._Ns
        # draw samples from prior density by drawing samples from transition density conditioned on
        # posterior samples drawn above. And the prior samples can be used in update step.
        proc_noi = multi_normal(0, Q_tilde, self._Ns, axis=0)
        for i in range(self._Ns):
            self._samples[i] = self._f(self._samples[i], u) + proc_noi[i]

        # compute prior_state and prior_cov
        self._prior_state = np.dot(self._weights, self._samples)
        self._prior_cov = 0
        for i in range(self._Ns):
            err = self._samples[i] - self._prior_state
            self._prior_cov += self._weights[i] * np.outer(err, err)
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

        self._stage = 1

    def update(self, z, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kw) > 0:
            if 'h' in kw: self._h = kw['h']
            if 'M' in kw: self._M = kw['M']
            if 'R' in kw: self._R = kw['R']
        # update weights to approximate the posterior density
        R_tilde = self._M @ self._R @ self._M.T
        for i in range(self._Ns):
            z_prior = self._h(self._samples[i])
            pdf = 1 / np.sqrt(lg.det(2 * np.pi * R_tilde))
            pdf *= np.exp(-0.5 * (z - z_prior) @ lg.inv(R_tilde) @ (z - z_prior))
            self._weights[i] = pdf
        self._weights[:] = self._weights / np.sum(self._weights)    # normalize
        # compute post_state and post_cov and the samples have been drawn in predict step
        self._post_state = np.dot(self._weights, self._samples)
        self._post_cov = 0
        for i in range(self._Ns):
            err = self._samples[i] - self._post_state
            self._post_cov += self._weights[i] * np.outer(err, err)
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

        self._len += 1
        self._stage = 0  # update finished

    def step(self, z, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict(u, **kw)
        self.update(z, **kw)
