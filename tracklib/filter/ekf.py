# -*- coding: utf-8 -*-
'''
This module includes first-order extended Kalman filter and
sceond-order extended Kalman filter.
'''
from __future__ import division, absolute_import, print_function


__all__ = ['EKFilter_1st', 'EKFilter_2ed']

import numpy as np
import scipy.linalg as lg
from .kfbase import KFBase
from tracklib.math import num_diff, num_diff_hessian


class EKFilter_1st(KFBase):
    '''
    First-order extended Kalman filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1, w_k-1)
    z_k = h_k(x_k, v_k)
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, f, h, Q, R, at=1):
        super().__init__()

        self._f = f
        self._h = h
        self._Q = Q
        self._R = R
        self._at = at

    def __str__(self):
        msg = 'Fisrt-order extended Kalman filter'
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

        x_dim = len(self._prior_state)
        w_dim = self._Q.shape[0]

        if len(kw) > 0:
            if 'f' in kw:
                self._f = kw['f']
            if 'Q' in kw:
                self._Q = kw['Q']

            if 'F' in kw:
                # Jacobian matrix provided by user for transition function
                F = kw['F']
            else:
                fx = lambda x: self._f(x, u, np.zeros(w_dim))
                F = num_diff(self._post_state, fx, x_dim)
            if 'L' in kw:
                L = kw['L']
            else:
                fw = lambda w: self._f(self._post_state, u, w)
                L = num_diff(np.zeros(w_dim), fw, x_dim)
        else:
            fx = lambda x: self._f(x, u, np.zeros(w_dim))
            F = num_diff(self._post_state, fx, x_dim)
            fw = lambda w: self._f(self._post_state, u, w)
            L = num_diff(np.zeros(w_dim), fw, x_dim)

        Q_tilde = L @ self._Q @ L.T
        self._prior_state = self._f(self._post_state, u, np.zeros(w_dim))
        self._prior_cov = self._at**2 * F @ self._post_cov @ F.T + Q_tilde
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

        self._stage = 1

    def update(self, z, it=0, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        x_dim = len(self._prior_state)
        z_dim = len(z)
        v_dim = self._R.shape[0]

        if len(kw) > 0:
            if 'h' in kw:
                self._h = kw['h']
            if 'R' in kw:
                self._R = kw['R']

            if 'H' in kw:
                # Jacobian matrix provided by user for measurement function
                H = kw['H']
                # If a user-defined Jacobian matrix is used, then iterated Kalman filter will not be used
                it = 0
            else:
                hx = lambda x: self._h(x, np.zeros(v_dim))
                H = num_diff(self._prior_state, hx, z_dim)
            if 'M' in kw:
                M = kw['M']
                it = 0
            else:
                hv = lambda v: self._h(self._prior_state, v)
                M = num_diff(np.zeros(v_dim), hv, z_dim)
        else:
            hx = lambda x: self._h(x, np.zeros(v_dim))
            H = num_diff(self._prior_state, hx, z_dim)
            hv = lambda v: self._h(self._prior_state, v)
            M = num_diff(np.zeros(v_dim), hv, z_dim)

        R_tilde = M @ self._R @ M.T
        z_prior = self._h(self._prior_state, np.zeros(v_dim))
        self._innov = z - z_prior
        self._innov_cov = H @ self._prior_cov @ H.T + R_tilde
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2
        self._gain = self._prior_cov @ H.T @ lg.inv(self._innov_cov)
        self._post_state = self._prior_state + self._gain @ self._innov
        # The Joseph-form covariance update is used for improved numerical
        temp = np.eye(x_dim) - self._gain @ H
        self._post_cov = temp @ self._prior_cov @ temp.T + self._gain @ R_tilde @ self._gain.T
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

        for _ in range(it):
            hx = lambda x: self._h(x, np.zeros(v_dim))
            H = num_diff(self._post_state, hx, z_dim)
            hv = lambda v: self._h(self._post_state, v)
            M = num_diff(np.zeros(v_dim), hv, z_dim)

            R_tilde = M @ self._R @ M.T
            z_prior = self._h(self._post_state, np.zeros(v_dim)) + H @ (self._prior_state - self._post_state)
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

    def step(self, z, u=None, it=0, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        self.predict(u, **kw)
        self.update(z, it=it, **kw)


class EKFilter_2ed(KFBase):
    '''
    Second-order extended Kalman filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1, w_k-1)
    z_k = h_k(x_k, v_k)
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, f, h, Q, R, at=1):
        super().__init__()
        self._at = at

        self._f = f
        self._h = h
        self._Q = Q
        self._R = R

    def __str__(self):
        msg = 'Second-order Kalman filter'
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

        x_dim = len(self._prior_state)
        w_dim = self._Q.shape[0]

        if len(kw) > 0:
            if 'f' in kw:
                self._f = kw['f']
            if 'Q' in kw:
                self._Q = kw['Q']

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
            if 'f_Hess' in kw:
                f_Hess = kw['f_Hess']
            else:
                fx = lambda x: self._f(x, u, np.zeros(w_dim))
                f_Hess = num_diff_hessian(self._post_state, fx, x_dim)
        else:
            fx = lambda x: self._f(x, u, np.zeros(w_dim))
            F = num_diff(self._post_state, fx, x_dim)
            f_Hess = num_diff_hessian(self._post_state, fx, x_dim)
            fw = lambda w: self._f(self._post_state, u, w)
            L = num_diff(np.zeros(w_dim), fw, x_dim)

        Q_tilde = L @ self._Q @ L.T
        self._prior_state = self._f(self._post_state, u, np.zeros(w_dim))
        quad = np.array([np.trace(f_Hess[:, :, i] @ self._post_cov) for i in range(x_dim)])
        self._prior_state += quad / 2
        self._prior_cov = self._at**2 * F @ self._post_cov @ F.T + Q_tilde
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

        self._stage = 1

    def update(self, z, it=0, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        x_dim = len(self._prior_state)
        z_dim = len(z)
        v_dim = self._R.shape[0]

        if len(kw) > 0:
            if 'h' in kw:
                self._h = kw['h']
            if 'R' in kw:
                self._R = kw['R']

            if 'H' in kw:
                H = kw['H']
                it = 0
            else:
                hx = lambda x: self._h(x, np.zeros(v_dim))
                H = num_diff(self._prior_state, hx, z_dim)
            if 'M' in kw:
                M = kw['M']
                it = 0
            else:
                hv = lambda v: self._h(self._prior_state, v)
                M = num_diff(np.zeros(v_dim), hv, z_dim)
            if 'h_Hess' in kw:
                h_Hess = kw['h_Hess']
            else:
                hx = lambda x: self._h(x, np.zeros(v_dim))
                h_Hess = num_diff_hessian(self._prior_state, hx, z_dim)
        else:
            hx = lambda x: self._h(x, np.zeros(v_dim))
            H = num_diff(self._prior_state, hx, z_dim)
            h_Hess = num_diff_hessian(self._prior_state, hx, z_dim)
            hv = lambda v: self._h(self._prior_state, v)
            M = num_diff(np.zeros(v_dim), hv, z_dim)

        R_tilde = M @ self._R @ M.T
        quad = np.array([np.trace(h_Hess[:, :, i] @ self._prior_cov) for i in range(z_dim)])
        z_prior = self._h(self._prior_state, np.zeros(v_dim)) + quad / 2
        self._innov = z - z_prior
        self._innov_cov = H @ self._prior_cov @ H.T + R_tilde
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2
        self._gain = self._prior_cov @ H.T @ lg.inv(self._innov_cov)
        self._post_state = self._prior_state + self._gain @ self._innov
        # The Joseph-form covariance update is used for improved numerical
        temp = np.eye(x_dim) - self._gain @ H
        self._post_cov = temp @ self._prior_cov @ temp.T + self._gain @ R_tilde @ self._gain.T
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

        for _ in range(it):
            hx = lambda x: self._h(x, np.zeros(v_dim))
            H = num_diff(self._post_state, hx, z_dim)
            h_Hess = num_diff_hessian(self._post_state, hx, z_dim)
            hv = lambda v: self._h(self._post_state, v)
            M = num_diff(np.zeros(v_dim), hv, z_dim)

            R_tilde = M @ self._R @ M.T
            quad = np.array([np.trace(h_Hess[:, :, i] @ self._post_cov) for i in range(z_dim)])
            z_prior = self._h(self._post_state, np.zeros(v_dim)) + H @ (self._prior_state - self._post_state) + quad / 2
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

    def step(self, z, u=None, it=0, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        self.predict(u, **kw)
        self.update(z, it=it, **kw)
