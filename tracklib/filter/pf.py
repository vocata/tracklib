# -*- coding: utf-8 -*-
'''
Particle filter
'''
from __future__ import division, absolute_import, print_function

__all__ = ['SIRPFilter']

import numpy as np
import scipy.linalg as lg
from .base import PFBase
from tracklib.utils import crndn, drnd


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
    def __init__(self, f, L, h, M, Q, R, Ns=50):
        super().__init__()

        self._f = f
        self._L = L
        self._h = h
        self._M = M
        self._Q = Q
        self._R = R
        self._Ns = Ns

    def __str__(self):
        msg = 'SIR particle filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        self._samples = crndn(state, cov, self._Ns, axis=0)
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
        proc_noi = crndn(0, Q_tilde, self._Ns, axis=0)
        for i in range(self._Ns):
            self._samples[i] = self._f(self._samples[i], u) + proc_noi[i]

        # update weights
        R_tilde = self._M @ self._R @ self._M.T
        v_dim = R_tilde.shape[0]
        for i in range(self._Ns):
            z_prior = self._h(self._samples[i])
            tmp = 1 / np.sqrt(lg.det(2 * np.pi * R_tilde))
            tmp *= np.exp(-0.5 * (z - z_prior) @ lg.inv(R_tilde) @ (z - z_prior))
            self._weights[i] = tmp
        self._weights[:] = self._weights / np.sum(self._weights)
        
        # resampling
        self._samples[:], _ = drnd(self._weights, self._Ns, self._samples)
        self._weights[:] = 1 / self._Ns

        self._len += 1
