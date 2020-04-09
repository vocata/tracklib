# -*- coding: utf-8 -*-
'''
Particle filter
'''
from __future__ import division, absolute_import, print_function

__all__ = ['SIRPFilter', 'RPFilter', 'EpanechnikovKernal', 'GuassianKernal']

import numpy as np
import scipy.linalg as lg
from .base import PFBase
from tracklib.utils import multi_normal, disc_random


class SIRPFilter(PFBase):
    '''
    Sampling importance resampling (SIR) filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1) + L_k-1*w_k-1
    z_k = h_k(x_k) + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k and v_k are additive noise
    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, f, L, h, M, Q, R, Ns, Neff, resample_alg='roulette'):
        super().__init__()

        self._f = f
        self._L = L
        self._h = h
        self._M = M
        self._Q = Q
        self._R = R
        self._Ns = Ns
        self._Neff = Neff
        self._resample_alg = resample_alg

    def __str__(self):
        msg = 'SIR particle filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        self._samples = multi_normal(state, cov, self._Ns, axis=0)
        self._weights = np.zeros(self._Ns) + 1 / self._Ns
        self._len = 0
        self._init = True

    def step(self, z, u=None, **kw):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kw) > 0:
            if 'f' in kw: self._f = kw['f']
            if 'L' in kw: self._L = kw['L']
            if 'Q' in kw: self._Q = kw['Q']
            if 'h' in kw: self._h = kw['h']
            if 'M' in kw: self._M = kw['M']
            if 'R' in kw: self._R = kw['R']

        # update samples
        Q_tilde = self._L @ self._Q @ self._L.T
        proc_noi = multi_normal(0, Q_tilde, self._Ns, axis=0)
        for i in range(self._Ns):
            self._samples[i] = self._f(self._samples[i], u) + proc_noi[i]

        # update weights
        R_tilde = self._M @ self._R @ self._M.T
        v_dim = R_tilde.shape[0]
        for i in range(self._Ns):
            z_prior = self._h(self._samples[i])
            pdf = 1 / np.sqrt(lg.det(2 * np.pi * R_tilde))
            pdf *= np.exp(-0.5 * (z - z_prior) @ lg.inv(R_tilde) @ (z - z_prior))
            self._weights[i] *= pdf
        self._weights[:] = self._weights / np.sum(self._weights)    # normalize

        # resample
        Neff = 1 / np.sum(self._weights**2)
        if Neff <= self._Neff:
            self._samples[:], _ = disc_random(self._weights, self._Ns, self._samples, alg=self._resample_alg)
            self._weights[:] = 1 / self._Ns

        self._len += 1


class RPFilter(PFBase):
    def __init__(self, f, L, h, M, Q, R, Ns, Neff, kernal, resample_alg='roulette'):
        super().__init__()

        self._f = f
        self._L = L
        self._h = h
        self._M = M
        self._Q = Q
        self._R = R
        self._Ns = Ns
        self._Neff = Neff
        self._kernal = kernal
        self._resample_alg = resample_alg

    def __str__(self):
        msg = 'Regularized particle filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        self._samples = multi_normal(state, cov, Ns=self._Ns, axis=0)
        self._weights = np.zeros(self._Ns) + 1 / self._Ns

        self._len = 0
        self._init = True

    def step(self, z, u=None, **kw):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kw) > 0:
            if 'f' in kw: self._f = kw['f']
            if 'L' in kw: self._L = kw['L']
            if 'Q' in kw: self._Q = kw['Q']
            if 'h' in kw: self._h = kw['h']
            if 'M' in kw: self._M = kw['M']
            if 'R' in kw: self._R = kw['R']

        # update samples
        Q_tilde = self._L @ self._Q @ self._L.T
        proc_noi = multi_normal(0, Q_tilde, Ns=self._Ns, axis=0)
        for i in range(self._Ns):
            self._samples[i] = self._f(self._samples[i], u) + proc_noi[i]

        # update weights
        R_tilde = self._M @ self._R @ self._M.T
        for i in range(self._Ns):
            z_prior = self._h(self._samples[i])
            pdf = 1 / np.sqrt(lg.det(2 * np.pi * R_tilde))
            pdf *= np.exp(-0.5 * (z - z_prior) @ lg.inv(R_tilde) @ (z - z_prior))
            self._weights[i] *= pdf
        self._weights[:] = self._weights / np.sum(self._weights)

        # regularization
        Neff = 1 / np.sum(self._weights**2)
        if Neff <= self._Neff:
            self._samples[:], self._weights[:] = self._kernal.resample(
                self._samples, self._weights, resample_alg=self._resample_alg)

        self._len += 1


class EpanechnikovKernal():
    def __init__(self, dim, Ns):
        vol = EpanechnikovKernal.unit_hypershpere_volumn(dim)
        n = dim + 4
        self.opt_bandwidth = ((8 * n * (2 * np.sqrt(np.pi))**dim / vol) / Ns)**(1 / n)
        self._dim = dim
        self._Ns = Ns

    def resample(self, samples, weights, resample_alg='roulette'):
        emp_mean = np.dot(weights, samples)
        emp_cov = 0
        for i in range(self._Ns):
            err = samples[i] - emp_mean
            emp_cov += weights[i] * np.outer(err, err)
        emp_cov = (emp_cov + emp_cov.T) / 2
        U, S, V = lg.svd(emp_cov)
        D = U @ np.diag(np.sqrt(S)) @ V.T

        sample, _ = disc_random(weights, self._Ns, samples, alg=resample_alg)
        sample = np.array(sample)
        weight = np.zeros_like(weights) + 1 / self._Ns

        # sample from beta distribution
        beta = np.random.beta(self._dim / 2, 2, self._Ns)
        # sample from a uniform distribution over unit sphere
        r = np.random.rand(self._Ns)
        r = r**(1 / self._dim)      # cdf: r^(1/n)
        theta = np.random.randn(self._dim, self._Ns)
        theta = theta / lg.norm(theta, axis=0)       # normalize random vector
        T = r * theta
        # sample from epanechnikov kernal
        eps = np.sqrt(beta) * T
        sample[:] = sample + self.opt_bandwidth * np.dot(D, eps).T

        return sample, weight

    @staticmethod
    def unit_hypershpere_volumn(dim):
        if dim % 2 == 0:
            vol = np.pi
            n = 2
            while n < dim:
                n += 2
                vol = 2 * np.pi * vol / n
        else:
            vol = 2
            n = 1
            while n < dim:
                n += 2
                vol = 2 * np.pi * vol / n
        return vol


class GuassianKernal():
    def __init__(self, dim, Ns):
        n = dim + 4
        self.opt_bandwidth = (4 / (dim + 2) / Ns)**(1 / n)
        self._dim = dim
        self._Ns = Ns

    def resample(self, samples, weights, resample_alg='roulette'):
        emp_mean = np.dot(weights, samples)
        emp_cov = 0
        for i in range(self._Ns):
            err = samples[i] - emp_mean
            emp_cov += weights[i] * np.outer(err, err)
        emp_cov = (emp_cov + emp_cov.T) / 2
        U, S, V = lg.svd(emp_cov)
        D = U @ np.diag(np.sqrt(S)) @ V.T

        sample, _ = disc_random(weights, self._Ns, samples, alg=resample_alg)
        sample = np.array(sample)
        weight = np.zeros_like(weights) + 1 / self._Ns

        eps = np.random.randn(self._dim, self._Ns)
        sample[:] = sample + self.opt_bandwidth * np.dot(D, eps).T

        return sample, weight