# -*- coding: utf-8 -*-
'''
Linear Kalman filter

REFERENCE:
[1]. D. Simon, "Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches," John Wiley and Sons, Inc., 2006.
'''
from __future__ import division, absolute_import, print_function


__all__ = ['KFilter', 'SeqKFilter']

import numpy as np
import scipy.linalg as lg
from .base import KFBase


class KFilter(KFBase):
    '''
    Standard linear Kalman filter, see[1]

    system model:
    x_k = F_k-1*x_k-1 + G_k-1*u_k-1 + L_k-1*w_k-1
    z_k = H_k*x_k + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, F, L, H, M, Q, R, G=None, at=1):
        super().__init__()

        self._F = F
        self._L = L
        self._H = H
        self._M = M
        self._Q = Q
        self._R = R
        self._G = G
        self._at = at

    def __str__(self):
        msg = 'Standard linear Kalman filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def _set_post_state(self, state):
        self._post_state[:] = state
    
    def _set_post_cov(self, cov):
        self._post_cov[:] = cov

    def init(self, state, cov):
        self._post_state = state
        self._post_cov = cov
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kw) > 0:
            if 'F' in kw: self._F = kw['F']
            if 'G' in kw: self._G = kw['G']
            if 'L' in kw: self._L = kw['L']
            if 'Q' in kw: self._Q = kw['Q']

        Q_tilde = self._L @ self._Q @ self._L.T
        ctl = 0 if u is None else self._G @ u
        self._prior_state = self._F @ self._post_state + ctl
        self._prior_cov = self._at**2 * self._F @ self._post_cov @ self._F.T + Q_tilde
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

        self._stage = 1  # predict finished

    def update(self, z, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        x_dim = len(self._post_state)

        if len(kw) > 0:
            if 'H' in kw: self._H = kw['H']
            if 'M' in kw: self._M = kw['M']
            if 'R' in kw: self._R = kw['R']

        R_tilde = self._M @ self._R @ self._M.T
        z_prior = self._H @ self._prior_state
        self._innov = z - z_prior
        self._innov_cov = self._H @ self._prior_cov @ self._H.T + R_tilde
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2
        self._gain = self._prior_cov @ self._H.T @ lg.inv(self._innov_cov)
        self._post_state = self._prior_state + self._gain @ self._innov
        # the Joseph-form covariance update is used for improving numerical
        temp = np.eye(x_dim) - self._gain @ self._H
        self._post_cov = temp @ self._prior_cov @ temp.T + self._gain @ R_tilde @ self._gain.T
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

        self._len += 1
        self._stage = 0  # update finished

    def step(self, z, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict(u, **kw)
        self.update(z, **kw)


class SeqKFilter(KFBase):
    '''
    Sequential linear Kalman filter, see[1]

    system model:
    x_k = F_k-1*x_k-1 + G_k-1*u_k-1 + L_k-1*w_k-1
    z_k = H_k*x_k + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, F, L, H, M, Q, R, G=None, at=1):
        super().__init__()

        self._F = F
        self._L = L
        self._H = H
        self._M = M
        self._Q = Q
        self._R = R
        self._G = G
        self._at = at
        R_tilde = self._M @ self._R @ self._M.T
        d, self._S = lg.eigh(R_tilde)
        self._D = np.diag(d)

    def __str__(self):
        msg = 'Sequential linear Kalman filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def _set_post_state(self, state):
        self._post_state[:] = state
    
    def _set_post_cov(self, cov):
        self._post_cov[:] = cov

    def init(self, state, cov):
        self._post_state = state
        self._post_cov = cov
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kw) > 0:
            if 'F' in kw: self._F = kw['F']
            if 'G' in kw: self._G = kw['G']
            if 'L' in kw: self._L = kw['L']
            if 'Q' in kw: self._Q = kw['Q']

        Q_tilde = self._L @ self._Q @ self._L.T
        ctl = 0 if u is None else self._G @ u
        self._prior_state = self._F @ self._post_state + ctl
        self._prior_cov = self._at**2 * self._F @ self._post_cov @ self._F.T + Q_tilde
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

        self._stage = 1  # predict finished

    def update(self, z, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        x_dim = len(self._post_state)
        z_dim = len(z)

        if len(kw) > 0:
            if 'H' in kw: self._H = kw['H']
            if 'M' in kw: self._M = kw['M']
            if 'R' in kw: self._R = kw['R']
            if 'M' in kw or 'R' in kw:
                R_tilde = self._M @ self._R @ self._M.T
                d, self._S = lg.eigh(R_tilde)
                self._D = np.diag(d)

        prior_state = self._prior_state
        post_cov = self._prior_cov
        H_tilde = self._S.T @ self._H
        z_tilde = self._S.T @ z
        for n in range(z_dim):
            H_n = H_tilde[n, :]
            z_n = z_tilde[n]
            r_n = self._D[n, n]

            z_prior = H_n @ prior_state
            innov = z_n - z_prior
            innov_cov = H_n @ post_cov @ H_n + r_n
            gain = (post_cov @ H_n) / innov_cov
            prior_state = prior_state + gain * innov
            # the Joseph-form covariance update is used for improving numerical
            temp = np.eye(x_dim) - np.outer(gain, H_n)
            post_cov = temp @ post_cov @ temp.T + r_n * np.outer(gain, gain)
            post_cov = (post_cov + post_cov.T) / 2
        self._post_state = prior_state
        self._post_cov = post_cov

        self._len += 1
        self._stage = 0

    def step(self, z, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict(u, **kw)
        self.update(z, **kw)
