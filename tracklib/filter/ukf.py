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
    def __init__(self, f, h, L, M, Q, R, kappa):
        super().__init__()

        self._f = f
        self._h = h
        self._L = L
        self._M = M
        self._Q = Q
        self._R = R
        self._kappa = kappa
        self._weight = None

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
        x_dim = state.shape[0]
        self._weight = np.zeros((x_dim, 1))
        self._weight[0] = self._kappa / (x_dim + self._kappa)
        self._weight[1:] = 1 / (2 * (x_dim + self._kappa))
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        x_dim = self._prior_state.shape[0]

        if 'f' in kw:
            self._f = kw['f']
        if 'Q' in kw:
            self._Q = kw['Q']

        d, v = lg.eigh((x_dim + self._kappa) * self._post_cov)
        # P = P_sqrt @ P_sqrt.T
        P_sqrt = v @ np.diag(np.sqrt(d))

        sigma_map = np.zeros((x_dim, 2 * x_dim + 1))
        sigma_map[:, 0] = self._f(self._post_state[:, 0], u)
        for i in range(x_dim):
            x_err = col(P_sqrt[:, i])
            sigma_map[:, i + 1] = self._f(self._post_state + x_err, u)
            sigma_map[:, x_dim + i + 1] = self._f(self._post_state - x_err, u)

