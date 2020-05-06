# -*- coding: utf-8 -*-
'''
Extended Kalman filter

REFERENCE:
[1]. D. Simon, "Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches," John Wiley and Sons, Inc., 2006.
'''
from __future__ import division, absolute_import, print_function


__all__ = ['EKFilterAN', 'EKFilterNAN']

import numpy as np
import scipy.linalg as lg
from functools import partial
from .base import KFBase
from tracklib.math import num_diff, num_diff_hessian


class EKFilterAN(KFBase):
    '''
    Additive extended Kalman filter, see[1]

    system model:
    x_k = f_k-1(x_k-1, u_k-1) + L_k-1*w_k-1
    z_k = h_k(x_k) + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, f, L, h, M, Q, R, xdim, zdim, fjac=None, hjac=None, fhes=None, hhes=None, order=1, it=0):
        super().__init__()

        self._f = lambda x, u: f(x, u)
        self._L = L.copy()
        self._h = lambda x: h(x)
        self._M = M.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._xdim = xdim
        self._zdim = zdim
        if fjac is None:
            def fjac(x, u):
                F = num_diff(x, partial(self._f, u=u), self._xdim)
                return F
        self._fjac = fjac
        if hjac is None:
            def hjac(x):
                H = num_diff(x, partial(self._h), self._zdim)
                return H
        self._hjac = hjac
        if fhes is None:
            def fhes(x, u):
                FH = num_diff_hessian(x, partial(self._f, u=u), self._xdim)
                return FH
        self._fhes = fhes
        if hhes is None:
            def hhes(x):
                HH = num_diff_hessian(x, self._h, self._zdim)
                return HH
        self._hhes = hhes
        if order == 1 or order == 2:
            self._order = order
        else:
            raise ValueError('order must be 1 or 2')
        self._it = it

    def __str__(self):
        msg = '%s-order additive noise extended Kalman filter' % ('First' if self._order == 1 else 'Second')
        return msg

    def __repr__(self):
        return self.__str__()

    def _set_state(self, state):
        self._state = state.copy()

    def _set_cov(self, cov):
        self._cov = cov.copy()

    def init(self, state, cov):
        self._state = state.copy()
        self._cov = cov.copy()
        self._init = True

    def predict(self, u=None, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'L' in kwargs: self._L[:] = kwargs['L']
            if 'Q' in kwargs: self._Q[:] = kwargs['Q']

        F = self._fjac(self._state, u)
        Q_tilde = self._L @ self._Q @ self._L.T
        self._state = self._f(self._state, u)
        self._cov = F @ self._cov @ F.T + Q_tilde
        self._cov = (self._cov + self._cov.T) / 2
        if self._order == 2:
            FH = self._fhes(self._state, u)
            quad = np.array([np.trace(FH[:, :, i] @ self._cov) for i in range(self._xdim)], dtype=float)
            self._state += quad / 2

    def correct(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'M' in kwargs: self._M[:] = kwargs['M']
            if 'R' in kwargs: self._R[:] = kwargs['R']

        prior_state = self._state
        prior_cov = self._cov

        H = self._hjac(prior_state)
        z_pred = self._h(prior_state)
        if self._order == 2:
            HH = self._hhes(prior_state)
            quad = np.array([np.trace(HH[:, :, i] @ prior_cov) for i in range(self._zdim)], dtype=float)
            z_pred += quad / 2
        innov = z - z_pred
        R_tilde = self._M @ self._R @ self._M.T
        S = H @ prior_cov @ H.T + R_tilde
        S = (S + S.T) / 2
        K = prior_cov @ H.T @ lg.inv(S)

        self._state = prior_state + K @ innov
        self._cov = prior_cov - K @ S @ K.T
        self._cov = (self._cov + self._cov.T) / 2

        for _ in range(self._it):
            H = self._hjac(self._state)
            z_pred = self._h(self._state) + H @ (prior_state - self._state)
            if self._order == 2:
                HH = self._hhes(self._state)
                quad = np.array([np.trace(HH[:, :, i] @ self._cov) for i in range(self._zdim)], dtype=float)
                z_pred += quad / 2
            innov = z - z_pred
            R_tilde = self._M @ self._R @ self._M.T
            S = H @ prior_cov @ H.T + R_tilde
            S = (S + S.T) / 2
            K = prior_cov @ H.T @ lg.inv(S)

            self._state = prior_state + K @ innov
            self._cov = prior_cov - K @ S @ K.T
            self._cov = (self._cov + self._cov.T) / 2

    def distance(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'M' in kwargs: self._M[:] = kwargs['M']
            if 'R' in kwargs: self._R[:] = kwargs['R']

        H = self._hjac(self._state)
        z_pred = self._h(self._state)
        if self._order == 2:
            HH = self._hhes(self._state)
            quad = np.array([np.trace(HH[:, :, i] @ self._cov) for i in range(self._zdim)], dtype=float)
            z_pred += quad / 2
        innov = z - z_pred
        R_tilde = self._M @ self._R @ self._M.T
        S = H @ self._state @ H.T + R_tilde
        S = (S + S.T) / 2
        d = innov @ lg.inv(S) @ innov + np.log(lg.det(S))

        return d
    
    def likelihood(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'M' in kwargs: self._M[:] = kwargs['M']
            if 'R' in kwargs: self._R[:] = kwargs['R']

        H = self._hjac(self._state)
        z_pred = self._h(self._state)
        if self._order == 2:
            HH = self._hhes(self._state)
            quad = np.array([np.trace(HH[:, :, i] @ self._cov) for i in range(self._zdim)], dtype=float)
            z_pred += quad / 2
        innov = z - z_pred
        R_tilde = self._M @ self._R @ self._M.T
        S = H @ self._state @ H.T + R_tilde
        S = (S + S.T) / 2
        pdf = np.exp(-innov @ lg.inv(S) @ innov / 2) / np.sqrt(lg.det(2 * np.pi * S))

        return pdf


class EKFilterNAN(KFBase):
    '''
    Nonadditive Extended Kalman filter, see[1]

    system model:
    x_k = f_k-1(x_k-1, u_k-1, w_k-1)
    z_k = h_k(x_k, v_k)
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, f, h, Q, R, xdim, zdim, fjac=None, hjac=None, fhes=None, hhes=None, order=1, it=0):
        super().__init__()

        self._f = lambda x, u, w: f(x, u, w)
        self._h = lambda x, v: h(x, v)
        self._Q = Q.copy()
        self._R = R.copy()
        self._xdim = xdim
        self._wdim = self._Q.shape[0]
        self._zdim = zdim
        self._vdim = self._R.shape[0]
        if fjac is None:
            def fjac(x, u, w):
                F = num_diff(x, partial(self._f, u=u, w=w), self._xdim)
                L = num_diff(w, partial(self._f, x, u), self._wdim)
                return F, L
        self._fjac = fjac
        if hjac is None:
            def hjac(x, v):
                H = num_diff(x, partial(self._h, v=v), self._zdim)
                M = num_diff(v, partial(self._h, x), self._vdim)
                return H, M
        self._hjac = hjac
        if fhes is None:
            def fhes(x, u, w):
                FH = num_diff_hessian(x, partial(self._f, u=u, w=w), self._xdim)
                return FH
        self._fhes = fhes
        if hhes is None:
            def hhes(x, v):
                HH = num_diff_hessian(x, partial(self._h, v=v), self._zdim)
                return HH
        self._hhes = hhes
        if order == 1 or order == 2:
            self._order = order
        else:
            raise ValueError('order must be 1 or 2')
        self._it = it

    def __str__(self):
        msg = '%s-order nonadditive noise extended Kalman filter' % ('First' if self._order == 1 else 'Second')
        return msg

    def __repr__(self):
        return self.__str__()

    def _set_state(self, state):
        self._state = state.copy()

    def _set_cov(self, cov):
        self._cov = cov.copy()

    def init(self, state, cov):
        self._state = state.copy()
        self._cov = cov.copy()
        self._init = True

    def predict(self, u=None, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if 'Q' in kwargs: self._Q[:] = kwargs['Q']

        F, L = self._fjac(self._state, u, np.zeros(self._wdim))
        Q_tilde = L @ self._Q @ L.T
        self._state = self._f(self._state, u, np.zeros(self._wdim))
        self._cov = F @ self._cov @ F.T + Q_tilde
        self._cov = (self._cov + self._cov.T) / 2
        if self._order == 2:
            FH = self._fhes(self._state, u, np.zeros(self._wdim))
            quad = np.array([np.trace(FH[:, :, i] @ self._cov) for i in range(self._xdim)], dtype=float)
            self._state += quad / 2

    def correct(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if 'R' in kwargs: self._R[:] = kwargs['R']

        prior_state = self._state
        prior_cov = self._cov

        H, M = self._hjac(prior_state, np.zeros(self._vdim))
        z_pred = self._h(prior_state, np.zeros(self._vdim))
        if self._order == 2:
            HH = self._hhes(prior_state, np.zeros(self._vdim))
            quad = np.array([np.trace(HH[:, :, i] @ prior_cov) for i in range(self._zdim)], dtype=float)
            z_pred += quad / 2
        innov = z - z_pred
        R_tilde = M @ self._R @ M.T
        S = H @ prior_cov @ H.T + R_tilde
        S = (S + S.T) / 2
        K = prior_cov @ H.T @ lg.inv(S)

        self._state = prior_state + K @ innov
        self._cov = prior_cov - K @ S @ K.T
        self._cov = (self._cov + self._cov.T) / 2

        for _ in range(self._it):
            H, M = self._hjac(self._state, np.zeros(self._vdim))
            z_pred = self._h(self._state, np.zeros(self._vdim)) + H @ (prior_state - self._state)
            if self._order == 2:
                HH = self._hhes(self._state, np.zeros(self._vdim))
                quad = np.array([np.trace(HH[:, :, i] @ self._cov) for i in range(self._zdim)], dtype=float)
                z_pred += quad / 2
            innov = z - z_pred
            R_tilde = M @ self._R @ M.T
            S = H @ prior_cov @ H.T + R_tilde
            S = (S + S.T) / 2
            K = prior_cov @ H.T @ lg.inv(S)

            self._state = prior_state + K @ innov
            self._cov = prior_cov - K @ S @ K.T
            self._cov = (self._cov + self._cov.T) / 2

    def distance(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if 'R' in kwargs: self._R[:] = kwargs['R']

        H, M = self._hjac(self._state, np.zeros(self._vdim))
        z_pred = self._h(self._state, np.zeros(self._vdim))
        if self._order == 2:
            HH = self._hhes(self._state, np.zeros(self._vdim))
            quad = np.array([np.trace(HH[:, :, i] @ self._cov) for i in range(self._zdim)], dtype=float)
            z_pred += quad / 2
        innov = z - z_pred
        R_tilde = M @ self._R @ M.T
        S = H @ self._cov @ H.T + R_tilde
        S = (S + S.T) / 2
        d = innov @ lg.inv(S) @ innov + np.log(lg.det(S))

        return d

    def likelihood(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if 'R' in kwargs: self._R[:] = kwargs['R']

        H, M = self._hjac(self._state, np.zeros(self._vdim))
        z_pred = self._h(self._state, np.zeros(self._vdim))
        if self._order == 2:
            HH = self._hhes(self._state, np.zeros(self._vdim))
            quad = np.array([np.trace(HH[:, :, i] @ self._cov) for i in range(self._zdim)], dtype=float)
            z_pred += quad / 2
        innov = z - z_pred
        R_tilde = M @ self._R @ M.T
        S = H @ self._cov @ H.T + R_tilde
        S = (S + S.T) / 2
        pdf = np.exp(-innov @ lg.inv(S) @ innov / 2) / np.sqrt(lg.det(2 * np.pi * S))

        return pdf
