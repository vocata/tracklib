# -*- coding: utf-8 -*-
'''
Linear Kalman filter

REFERENCE:
[1]. D. Simon, "Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches," John Wiley and Sons, Inc., 2006.
'''
from __future__ import division, absolute_import, print_function


__all__ = ['KFilter']

import numpy as np
import scipy.linalg as lg
from .base import FilterBase


class KFilter(FilterBase):
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

        self._F = F.copy()
        self._L = L.copy()
        self._H = H.copy()
        self._M = M.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        if G is None:
            self._G = G
        else:
            self._G = G.copy()
        self._at = at   # attenuation factor

    def __str__(self):
        msg = 'Standard linear Kalman filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        self._state = state.copy()
        self._cov = cov.copy()
        self._init = True

    def reset(self, state, cov):
        self._state = state.copy()
        self._cov = cov.copy()

    def predict(self, u=None, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'F' in kwargs: self._F[:] = kwargs['F']
            if 'G' in kwargs: self._G[:] = kwargs['G']
            if 'L' in kwargs: self._L[:] = kwargs['L']
            if 'Q' in kwargs: self._Q[:] = kwargs['Q']

        Q_tilde = self._L @ self._Q @ self._L.T
        ctl = 0 if u is None else self._G @ u
        self._state = self._F @ self._state + ctl
        self._cov = self._at**2 * self._F @ self._cov @ self._F.T + Q_tilde
        self._cov = (self._cov + self._cov.T) / 2

    def correct(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'H' in kwargs: self._H[:] = kwargs['H']
            if 'M' in kwargs: self._M[:] = kwargs['M']
            if 'R' in kwargs: self._R[:] = kwargs['R']

        R_tilde = self._M @ self._R @ self._M.T
        innov = z - self._H @ self._state
        S = self._H @ self._cov @ self._H.T + R_tilde
        S = (S + S.T) / 2
        K = self._cov @ self._H.T @ lg.inv(S)

        self._state = self._state + K @ innov
        self._cov = self._cov - K @ S @ K.T
        self._cov = (self._cov + self._cov.T) / 2

    def correct_JPDA(self, zs, probs, **kwargs):
        z_len = len(zs)

        Hs = kwargs['H'] if 'H' in kwargs else [self._H] * z_len
        Ms = kwargs['M'] if 'M' in kwargs else [self._M] * z_len
        Rs = kwargs['R'] if 'R' in kwargs else [self._R] * z_len

        state_item = 0.0
        cov_item1, cov_item2 = 0.0, 0.0
        
        for i in range(z_len):
            S = Hs[i] @ self._cov @ Hs[i].T + Ms[i] @ Rs[i] @ Ms[i].T
            S = (S + S.T) / 2
            K = self._cov @ Hs[i] @ lg.inv(S)

            innov = zs[i] - Hs[i] @ self._state
            incre = np.dot(K, innov)
            state_item += probs[i] * incre
            cov_item1 += probs[i] * self._cov - K @ S @ K.T
            cov_item2 += probs[i] * np.outer(incre, incre)

        self._state = self._state + state_item
        self._cov = (1 - np.sum(probs)) * self._cov + cov_item1 - (cov_item2 - np.outer(state_item, state_item))
        self._cov = (self._cov + self._cov.T) / 2

    def distance(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        H = kwargs['H'] if 'H' in kwargs else self._H
        M = kwargs['M'] if 'M' in kwargs else self._M
        R = kwargs['R'] if 'R' in kwargs else self._R

        R_tilde = M @ R @ M.T
        innov = z - H @ self._state
        S = H @ self._cov @ H.T + R_tilde
        S = (S + S.T) / 2
        d = innov @ lg.inv(S) @ innov + np.log(lg.det(S))

        return d
        

    def likelihood(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        H = kwargs['H'] if 'H' in kwargs else self._H
        M = kwargs['M'] if 'M' in kwargs else self._M
        R = kwargs['R'] if 'R' in kwargs else self._R

        R_tilde = M @ R @ M.T
        innov = z - H @ self._state
        S = H @ self._cov @ H.T + R_tilde
        S = (S + S.T) / 2
        pdf = 1 / np.sqrt(lg.det(2 * np.pi * S))
        pdf *= np.exp(-innov @ lg.inv(S) @ innov / 2)

        return pdf
