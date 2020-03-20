# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
from .kfbase import KFBase
from ..utils import col

__all__ = ['UKFilter']


class UKFilter(KFBase):
    '''
    Standard unscented Kalman filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1) + L_k-1*w_k-1
    z_k = h_k(x_k) + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, f, L, h, M, Q, R, kappa=0):
        super().__init__()

        self._f = f
        self._L = L
        self._h = h
        self._M = M
        self._Q = Q
        self._R = R
        self._kappa = kappa

    def __str__(self):
        msg = 'Stardand unscented Kalman filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        self._prior_state = state
        self._prior_cov = cov
        self._post_state = state
        self._post_cov = cov
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        x_dim = self._prior_state.shape[0]

        if len(kw) > 0:
            if 'f' in kw: self._f = kw['f']
            if 'L' in kw: self._L = kw['L']
            if 'Q' in kw: self._Q = kw['Q']

        d, v = lg.eigh((x_dim + self._kappa) * self._post_cov)
        # definition of square root of matrix: P = P_sqrt @ P_sqrt.T
        cov_sqrt = v @ np.diag(np.sqrt(d))

        weight_zero = self._kappa / (x_dim + self._kappa)
        weight = 1 / (2 * (x_dim + self._kappa))
        self._prior_state = weight_zero * self._f(self._post_state, u)
        self.__f_map = [None] * (2 * x_dim + 1)
        self.__f_map[0] = self._prior_state
        for i in range(1, x_dim + 1):
            sigma_err = col(cov_sqrt[:, i - 1])
            self.__f_map[i] = self._f(self._post_state + sigma_err, u)
            self.__f_map[x_dim + i] = self._f(self._post_state - sigma_err, u)
            self._prior_state += weight * self.__f_map[i]
            self._prior_state += weight * self.__f_map[x_dim + i]

        state_err = self.__f_map[0] - self._prior_state
        self._prior_cov = weight_zero * state_err @ state_err.T
        for i in range(1, x_dim + 1):
            state_err = self.__f_map[i] - self._prior_state
            self._prior_cov += weight * state_err @ state_err.T
            state_err = self.__f_map[x_dim + i] - self._prior_state
            self._prior_cov += weight * state_err @ state_err.T
        self._prior_cov += self._L @ self._Q @ self._L.T
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2
        self._stage = 1

    def update(self, z, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        x_dim = self._prior_state.shape[0]

        if len(kw) > 0:
            if 'h' in kw: self._h = kw['h']
            if 'M' in kw: self._M = kw['M']
            if 'R' in kw: self._R = kw['R']

        d, v = lg.eigh((x_dim + self._kappa) * self._prior_cov)
        cov_sqrt = v @ np.diag(np.sqrt(d))
        weight_zero = self._kappa / (x_dim + self._kappa)
        weight = 1 / (2 * (x_dim + self._kappa))
        h_map = [None] * (2 * x_dim + 1)
        z_prior = weight_zero * self._h(self._prior_state)
        h_map[0] = z_prior
        for i in range(1, x_dim + 1):
            sigma_err = col(cov_sqrt[:, i - 1])
            h_map[i] = self._h(self._prior_state + sigma_err)
            h_map[x_dim + i] = self._h(self._prior_state - sigma_err)
            z_prior += weight * h_map[i]
            z_prior += weight * h_map[x_dim + i]

        state_err = self.__f_map[0] - self._prior_state
        z_err = h_map[0] - z_prior
        self._innov_cov = weight_zero * z_err @ z_err.T
        xz_cov = weight_zero * state_err @ z_err.T
        for i in range(1, x_dim + 1):
            state_err = self.__f_map[i] - self._prior_state
            z_err = h_map[i] - z_prior
            self._innov_cov += weight * z_err @ z_err.T
            xz_cov + weight * state_err @ z_err.T
            state_err = self.__f_map[x_dim + i] - self._prior_state
            z_err = h_map[x_dim + i] - z_prior
            self._innov_cov += weight * z_err @ z_err.T
            xz_cov + weight * state_err @ z_err.T
        self._innov_cov += self._M @ self._R @ self._M.T
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2

        self._innov = z - z_prior
        self._gain = xz_cov @ lg.inv(self._innov_cov)
        self._post_state = self._prior_state + self._gain @ self._innov
        self._post_cov = self._prior_cov - self._gain @ self._innov_cov @ self._gain.T

        self._len += 1
        self._stage = 0  # update finished

    def step(self, z, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        self.predict(u, **kw)
        self.update(z, **kw)
