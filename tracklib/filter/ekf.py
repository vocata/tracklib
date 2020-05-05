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
        self._wdim = self._Q.shape[0]
        self._zdim = zdim
        self._vdim = self._R.shape[0]
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

    def _set_post_state(self, state):
        self._post_state = state.copy()

    def _set_post_cov(self, cov):
        self._post_cov = cov.copy()

    def init(self, state, cov):
        self._post_state = state.copy()
        self._post_cov = cov.copy()
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self, u=None, **kwargs):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'L' in kwargs: self._L[:] = kwargs['L']
            if 'Q' in kwargs: self._Q[:] = kwargs['Q']

        F = self._fjac(self._post_state, u)
        Q_tilde = self._L @ self._Q @ self._L.T
        self._prior_cov = F @ self._post_cov @ F.T + Q_tilde
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2
        self._prior_state = self._f(self._post_state, u)
        if self._order == 2:
            FH = self._fhes(self._post_state, u)
            quad = np.array([np.trace(FH[:, :, i] @ self._post_cov) for i in range(self._xdim)], dtype=float)
            self._prior_state += quad / 2

        self._stage = 1

    def update(self, z, **kwargs):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'M' in kwargs: self._M[:] = kwargs['M']
            if 'R' in kwargs: self._R[:] = kwargs['R']

        H = self._hjac(self._prior_state)
        R_tilde = self._M @ self._R @ self._M.T
        z_prior = self._h(self._prior_state)
        if self._order == 2:
            HH = self._hhes(self._prior_state)
            quad = np.array([np.trace(HH[:, :, i] @ self._prior_cov) for i in range(self._zdim)], dtype=float)
            z_prior += quad / 2
        self._innov = z - z_prior
        self._innov_cov = H @ self._prior_cov @ H.T + R_tilde
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2
        self._gain = self._prior_cov @ H.T @ lg.inv(self._innov_cov)
        self._post_state = self._prior_state + self._gain @ self._innov
        # the Joseph-form covariance update is used for improving numerical
        temp = np.eye(self._xdim) - self._gain @ H
        self._post_cov = temp @ self._prior_cov @ temp.T + self._gain @ R_tilde @ self._gain.T
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

        for _ in range(self._it):
            H = self._hjac(self._post_state)
            R_tilde = self._M @ self._R @ self._M.T
            z_prior = self._h(self._post_state) + H @ (self._prior_state - self._post_state)
            if self._order == 2:
                HH = self._hhes(self._post_state)
                quad = np.array([np.trace(HH[:, :, i] @ self._post_cov) for i in range(self._zdim)], dtype=float)
                z_prior += quad / 2
            self._innov = z - z_prior
            self._innov_cov = H @ self._prior_cov @ H.T + R_tilde
            self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2
            self._gain = self._prior_cov @ H.T @ lg.inv(self._innov_cov)
            self._post_state = self._prior_state + self._gain @ self._innov
            temp = np.eye(self._xdim) - self._gain @ H
            self._post_cov = temp @ self._prior_cov @ temp.T + self._gain @ R_tilde @ self._gain.T
            self._post_cov = (self._post_cov + self._post_cov.T) / 2

        self._len += 1
        self._stage = 0

    def step(self, z, u=None, **kwargs):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict(u, **kwargs)
        self.update(z, **kwargs)


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

    def _set_post_state(self, state):
        self._post_state = state.copy()

    def _set_post_cov(self, cov):
        self._post_cov = cov.copy()

    def init(self, state, cov):
        self._post_state = state.copy()
        self._post_cov = cov.copy()
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self, u=None, **kwargs):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if 'Q' in kwargs: self._Q[:] = kwargs['Q']

        F, L = self._fjac(self._post_state, u, np.zeros(self._wdim))
        Q_tilde = L @ self._Q @ L.T
        self._prior_cov = F @ self._post_cov @ F.T + Q_tilde
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2
        self._prior_state = self._f(self._post_state, u, np.zeros(self._wdim))
        if self._order == 2:
            FH = self._fhes(self._post_state, u, np.zeros(self._wdim))
            quad = np.array([np.trace(FH[:, :, i] @ self._post_cov) for i in range(self._xdim)], dtype=float)
            self._prior_state += quad / 2

        self._stage = 1

    def update(self, z, **kwargs):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        if 'R' in kwargs: self._R[:] = kwargs['R']

        H, M = self._hjac(self._prior_state, np.zeros(self._vdim))
        R_tilde = M @ self._R @ M.T
        z_prior = self._h(self._prior_state, np.zeros(self._vdim))
        if self._order == 2:
            HH = self._hhes(self._prior_state, np.zeros(self._vdim))
            quad = np.array([np.trace(HH[:, :, i] @ self._prior_cov) for i in range(self._zdim)], dtype=float)
            z_prior += quad / 2
        self._innov = z - z_prior
        self._innov_cov = H @ self._prior_cov @ H.T + R_tilde
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2
        self._gain = self._prior_cov @ H.T @ lg.inv(self._innov_cov)
        self._post_state = self._prior_state + self._gain @ self._innov
        # the Joseph-form covariance update is used for improving numerical
        temp = np.eye(self._xdim) - self._gain @ H
        self._post_cov = temp @ self._prior_cov @ temp.T + self._gain @ R_tilde @ self._gain.T
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

        for _ in range(self._it):
            H, M = self._hjac(self._post_state, np.zeros(self._vdim))
            R_tilde = M @ self._R @ M.T
            z_prior = self._h(self._post_state, np.zeros(self._vdim)) + H @ (self._prior_state - self._post_state)
            if self._order == 2:
                HH = self._hhes(self._post_state, np.zeros(self._vdim))
                quad = np.array([np.trace(HH[:, :, i] @ self._post_cov) for i in range(self._zdim)], dtype=float)
                z_prior += quad / 2
            self._innov = z - z_prior
            self._innov_cov = H @ self._prior_cov @ H.T + R_tilde
            self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2
            self._gain = self._prior_cov @ H.T @ lg.inv(self._innov_cov)
            self._post_state = self._prior_state + self._gain @ self._innov
            temp = np.eye(self._xdim) - self._gain @ H
            self._post_cov = temp @ self._prior_cov @ temp.T + self._gain @ R_tilde @ self._gain.T
            self._post_cov = (self._post_cov + self._post_cov.T) / 2

        self._len += 1
        self._stage = 0

    def step(self, z, u=None, **kwargs):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict(u, **kwargs)
        self.update(z, **kwargs)
