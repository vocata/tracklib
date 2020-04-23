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
    def __init__(self, f, L, h, M, Q, R, order=1, it=0):
        super().__init__()

        self._f = f
        self._L = L
        self._h = h
        self._M = M
        self._Q = Q
        self._R = R
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

        x_dim = len(self._post_state)

        if len(kw) > 0:
            if 'f' in kw: self._f = kw['f']
            if 'L' in kw: self._L = kw['L']
            if 'Q' in kw: self._Q = kw['Q']

            if 'F' in kw:
                F = kw['F']
            else:
                fx = lambda x: self._f(x, u)
                F = num_diff(self._post_state, fx, x_dim)
            if self._order == 2:
                if 'FH' in kw:
                    FH = kw['FH']   # Hessian matrix of f
                else:
                    fx = lambda x: self._f(x, u)
                    FH = num_diff_hessian(self._post_state, fx, x_dim)
        else:
            fx = lambda x: self._f(x, u) 
            F = num_diff(self._post_state, fx, x_dim)
            if self._order == 2:
                FH = num_diff_hessian(self._post_state, fx, x_dim)

        Q_tilde = self._L @ self._Q @ self._L.T
        self._prior_state = self._f(self._post_state, u)
        if self._order == 2:
            quad = np.array([np.trace(FH[:, :, i] @ self._post_cov) for i in range(x_dim)])
            self._prior_state += quad / 2
        self._prior_cov = F @ self._post_cov @ F.T + Q_tilde
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

        self._stage = 1

    def update(self, z, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        x_dim = len(self._post_state)
        z_dim = len(z)

        if len(kw) > 0:
            if 'h' in kw: self._h = kw['h']
            if 'M' in kw: self._M = kw['M']
            if 'R' in kw: self._R = kw['R']

            if 'H' in kw:
                H = kw['H']
                self._it = 0    # if H is given, disable iterated EKF, the same as HH
            else:
                hx = lambda x: self._h(x)
                H = num_diff(self._prior_state, hx, z_dim)
            if self._order == 2:
                if 'HH' in kw:
                    HH = kw['HH']   # Hessian matrix of h
                    self._it = 0
                else:
                    hx = lambda x: self._h(x)
                    HH = num_diff_hessian(self._prior_state, hx, z_dim)
        else:
            hx = lambda x: self._h(x)
            H = num_diff(self._prior_state, hx, z_dim)
            if self._order == 2:
                HH = num_diff_hessian(self._prior_state, hx, z_dim)

        R_tilde = self._M @ self._R @ self._M.T
        z_prior = self._h(self._prior_state)
        if self._order == 2:
            quad = np.array([np.trace(HH[:, :, i] @ self._prior_cov) for i in range(z_dim)])
            z_prior += quad / 2
        self._innov = z - z_prior
        self._innov_cov = H @ self._prior_cov @ H.T + R_tilde
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2
        self._gain = self._prior_cov @ H.T @ lg.inv(self._innov_cov)
        self._post_state = self._prior_state + self._gain @ self._innov
        # the Joseph-form covariance update is used for improved numerical
        temp = np.eye(x_dim) - self._gain @ H
        self._post_cov = temp @ self._prior_cov @ temp.T + self._gain @ R_tilde @ self._gain.T
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

        for _ in range(self._it):
            hx = lambda x: self._h(x)
            H = num_diff(self._post_state, hx, z_dim)
            if self._order == 2:
                HH = num_diff_hessian(self._post_state, hx, z_dim)

            R_tilde = self._M @ self._R @ self._M.T
            z_prior = self._h(self._post_state) + H @ (self._prior_state - self._post_state)
            if self._order == 2:
                quad = np.array([np.trace(HH[:, :, i] @ self._post_cov) for i in range(z_dim)])
                z_prior += quad / 2
            self._innov = z - z_prior
            self._innov_cov = H @ self._prior_cov @ H.T + R_tilde
            self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2
            self._gain = self._prior_cov @ H.T @ lg.inv(self._innov_cov)
            self._post_state = self._prior_state + self._gain @ self._innov
            temp = np.eye(x_dim) - self._gain @ H
            self._post_cov = temp @ self._prior_cov @ temp.T + self._gain @ R_tilde @ self._gain.T
            self._post_cov = (self._post_cov + self._post_cov.T) / 2

        self._len += 1
        self._stage = 0

    def step(self, z, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict(u, **kw)
        self.update(z, **kw)


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
    def __init__(self, f, h, Q, R, order=1, it=0):
        super().__init__()

        self._f = f
        self._h = h
        self._Q = Q
        self._R = R
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

        x_dim = len(self._post_state)
        w_dim = self._Q.shape[0]

        if len(kw) > 0:
            if 'f' in kw: self._f = kw['f']
            if 'Q' in kw: self._Q = kw['Q']

            if 'F' in kw:
                F = kw['F']
            else:
                fx = lambda x: self._f(x, u, np.zeros(w_dim))
                F = num_diff(self._post_state, fx, x_dim)
            if 'L' in kw:
                L = kw['L']
            else:
                fw = lambda w: self._f(self._post_state, u, w)
                L = num_diff(np.zeros(w_dim), fw, x_dim)
            if self._order == 2:
                if 'FH' in kw:
                    FH = kw['FH']   # Hessian matrix of f
                else:
                    fx = lambda x: self._f(x, u, np.zeros(w_dim))
                    FH = num_diff_hessian(self._post_state, fx, x_dim)
        else:
            fx = lambda x: self._f(x, u, np.zeros(w_dim))
            F = num_diff(self._post_state, fx, x_dim)
            fw = lambda w: self._f(self._post_state, u, w)
            L = num_diff(np.zeros(w_dim), fw, x_dim)
            if self._order == 2:
                FH = num_diff_hessian(self._post_state, fx, x_dim)

        Q_tilde = L @ self._Q @ L.T
        self._prior_state = self._f(self._post_state, u, np.zeros(w_dim))
        if self._order == 2:
            quad = np.array([np.trace(FH[:, :, i] @ self._post_cov) for i in range(x_dim)])
            self._prior_state += quad / 2
        self._prior_cov = F @ self._post_cov @ F.T + Q_tilde
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

        self._stage = 1

    def update(self, z, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        x_dim = len(self._post_state)
        z_dim = len(z)
        v_dim = self._R.shape[0]

        if len(kw) > 0:
            if 'h' in kw: self._h = kw['h']
            if 'R' in kw: self._R = kw['R']

            if 'H' in kw:
                H = kw['H']
                self._it = 0
            else:
                hx = lambda x: self._h(x, np.zeros(v_dim))
                H = num_diff(self._prior_state, hx, z_dim)
            if 'M' in kw:
                M = kw['M']
                self._it = 0
            else:
                hv = lambda v: self._h(self._prior_state, v)
                M = num_diff(np.zeros(v_dim), hv, z_dim)
            if self._order == 2:
                if 'HH' in kw:
                    HH = kw['HH']   # Hessian matrix of h
                    self._it = 0
                else:
                    hx = lambda x: self._h(x, np.zeros(v_dim))
                    HH = num_diff_hessian(self._prior_state, hx, z_dim)
        else:
            hx = lambda x: self._h(x, np.zeros(v_dim))
            H = num_diff(self._prior_state, hx, z_dim)
            hv = lambda v: self._h(self._prior_state, v)
            M = num_diff(np.zeros(v_dim), hv, z_dim)
            if self._order == 2:
                HH = num_diff_hessian(self._prior_state, hx, z_dim)

        R_tilde = M @ self._R @ M.T
        z_prior = self._h(self._prior_state, np.zeros(v_dim))
        if self._order == 2:
            quad = np.array([np.trace(HH[:, :, i] @ self._prior_cov) for i in range(z_dim)])
            z_prior += quad / 2
        self._innov = z - z_prior
        self._innov_cov = H @ self._prior_cov @ H.T + R_tilde
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2
        self._gain = self._prior_cov @ H.T @ lg.inv(self._innov_cov)
        self._post_state = self._prior_state + self._gain @ self._innov
        # the Joseph-form covariance update is used for improved numerical
        temp = np.eye(x_dim) - self._gain @ H
        self._post_cov = temp @ self._prior_cov @ temp.T + self._gain @ R_tilde @ self._gain.T
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

        for _ in range(self._it):
            hx = lambda x: self._h(x, np.zeros(v_dim))
            H = num_diff(self._post_state, hx, z_dim)
            hv = lambda v: self._h(self._post_state, v)
            M = num_diff(np.zeros(v_dim), hv, z_dim)
            if self._order == 2:
                HH = num_diff_hessian(self._post_state, hx, z_dim)

            R_tilde = M @ self._R @ M.T
            z_prior = self._h(self._post_state, np.zeros(v_dim)) + H @ (self._prior_state - self._post_state)
            if self._order == 2:
                quad = np.array([np.trace(HH[:, :, i] @ self._post_cov) for i in range(z_dim)])
                z_prior += quad / 2
            self._innov = z - z_prior
            self._innov_cov = H @ self._prior_cov @ H.T + R_tilde
            self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2
            self._gain = self._prior_cov @ H.T @ lg.inv(self._innov_cov)
            self._post_state = self._prior_state + self._gain @ self._innov
            temp = np.eye(x_dim) - self._gain @ H
            self._post_cov = temp @ self._prior_cov @ temp.T + self._gain @ R_tilde @ self._gain.T
            self._post_cov = (self._post_cov + self._post_cov.T) / 2

        self._len += 1
        self._stage = 0

    def step(self, z, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict(u, **kw)
        self.update(z, **kw)
