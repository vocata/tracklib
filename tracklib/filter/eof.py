# -*- coding: utf-8 -*-
'''
Extended object tracker

REFERENCE:
[1]. 
'''
from __future__ import division, absolute_import, print_function


__all__ = ['KochEOFilter', 'FeldmannEOFilter', 'LanEOFilter']

import numpy as np
import scipy.linalg as lg
from .base import EOFilterBase


class KochEOFilter(EOFilterBase):
    '''
    Extended object particle filter using Koch approach
    '''
    def __init__(self, F, H, D, interval, tau, dim=2):
        self._F = F.copy()
        self._H = H.copy()
        self._D = D.copy()
        self._at = np.exp(-interval / tau)      # attenuation factor
        self._dim = dim

    def init(self, state, cov, df, extension):
        self._df = df
        self._scale = extension * (df - self._dim - 1)
        self._single_cov = cov.copy()

        self._state = state.copy()
        self._cov = np.kron(extension, cov)
        self._ext = extension.copy()
        self._init = True

    def predict(self):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        # predict inverse wishart parameters
        df = self._df
        self._df = self._at * self._df
        w = (self._df - self._dim - 1) / (df - self._dim - 1)
        self._scale = w * self._scale

        # predict joint state
        self._ext = self._scale / (self._df - self._dim - 1) * 2
        self._single_cov = self._F @ self._single_cov @ self._F.T + self._D
        self._single_cov = (self._single_cov + self._single_cov.T) / 2
        df_tilde = self._df + len(self._state) // self._dim + len(self._state)
        self._cov = np.kron(self._ext, self._single_cov) / (df_tilde - 2)
        F_tilde = np.kron(np.eye(self._dim), self._F)
        self._state = np.dot(F_tilde, self._state)

        return self._state, self._cov, self._ext

    def correct(self, zs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        n = len(zs)

        z_mean = np.mean(zs, axis=0)
        eps = z_mean - np.dot(np.kron(np.eye(self._dim), self._H), self._state)
        z_center = zs - z_mean
        Z = np.dot(z_center.T, z_center)
        S = self._H @ self._single_cov @ self._H.T + 1 / n
        S = (S + S.T) / 2
        S_inv = lg.inv(S)
        K = self._single_cov @ self._H.T @ S_inv
        N = S_inv * np.outer(eps, eps)

        # correct inverse wishart parameters
        self._df += n
        self._scale += N + Z

        # correct joint state
        self._ext = self._scale / (self._df - self._dim - 1) * 2
        self._single_cov -= K @ S @ K.T
        self._single_cov = (self._single_cov + self._single_cov.T) / 2
        df_tilde = self._df + len(self._state) // self._dim + len(self._state)
        self._cov = np.kron(self._ext, self._single_cov) / (df_tilde - 2)
        K_tilde = np.kron(np.eye(self._dim), K)
        self._state += np.dot(K_tilde, eps)

        return self._state, self._cov, self._ext

    def distance(self, zs, **kwargs):
        return super().distance(zs, **kwargs)

    def likelihood(self, zs, **kwargs):
        return super().likelihood(zs, **kwargs)


class FeldmannEOFilter(EOFilterBase):
    '''
    Extended object particle filter using Feldmann approach
    '''
    def __init__(self, F, H, Q, R, interval, tau, dim=2):
        self._F = F.copy()
        self._H = H.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._at = np.exp(-interval / tau)      # attenuation
        self._dim = dim

    def init(self, state, cov, df, extension):
        self._df = df - self._dim - 1

        self._state = state.copy()
        self._cov = cov.copy()
        self._ext = extension.copy()
        self._init = True

    def predict(self):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        self._state = np.dot(self._F, self._state)
        self._cov = self._F @ self._cov @ self._F.T + self._Q
        self._cov = (self._cov + self._cov.T) / 2
        self._ext = self._ext
        self._df = 2 + self._at * (self._df - 2)

        return self._state, self._cov, self._ext

    def correct(self, zs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        n = len(zs)

        z_mean = np.mean(zs, axis=0)
        eps = z_mean - np.dot(self._H, self._state)
        z_center = zs - z_mean
        Z = np.dot(z_center.T, z_center)
        Y = self._ext / 4 + self._R
        S = self._H @ self._cov @ self._H.T + Y / n
        S = (S + S.T) / 2
        X_chol = lg.cholesky(self._ext, lower=True)
        S_chol = lg.inv(lg.cholesky(S, lower=True))
        Y_chol = lg.inv(lg.cholesky(Y, lower=True))
        N = np.outer(eps, eps)
        N_hat = X_chol @ S_chol @ N @ S_chol.T @ X_chol.T
        Z_hat = X_chol @ Y_chol @ Z @ Y_chol.T @ X_chol.T
        df = self._df
        self._df += n
        self._ext = (df * self._ext + N_hat + Z_hat) / self._df

        K = self._cov @ self._H.T @ lg.inv(S)
        self._state += K @ eps
        self._cov -= K @ S @ K.T
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov, self._ext

    def distance(self, zs, **kwargs):
        return super().distance(zs, **kwargs)

    def likelihood(self, zs, **kwargs):
        return super().likelihood(zs, **kwargs)


class LanEOFilter(EOFilterBase):
    '''
    Extended object particle filter using Lan approach
    '''
    def __init__(self, F, H, D, R, delta, dim=2):
        self._F = F.copy()
        self._H = H.copy()
        self._D = D.copy()
        self._R = R.copy()
        self._delta = delta
        self._dim = dim

    def init(self, state, cov, df, extension):
        self._df = df
        self._scale = extension * (df - 2 * self._dim - 2)
        self._single_cov = cov.copy()

        self._state = state.copy()
        self._cov = np.kron(extension, cov)
        self._ext = extension.copy()
        self._init = True

    def predict(self):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        # predict inverse wishart parameters
        lamb = self._df - 2 * self._dim - 2
        self._df = 2 * self._delta * (lamb + 1) * (lamb - 1) * (lamb - 2) / lamb**2 / (lamb + self._delta) + 2 * self._dim + 4
        self._scale = (self._df - 2 * self._dim - 2) / lamb * self._scale

        # predict joint state
        self._ext = self._scale / (self._df - 2 * self._dim - 2)
        self._single_cov = self._F @ self._single_cov @ self._F.T + self._D
        self._single_cov = (self._single_cov + self._single_cov.T) / 2
        self._cov = np.kron(self._ext, self._single_cov)
        F_tilde = np.kron(np.eye(self._dim), self._F)
        self._state = np.dot(F_tilde, self._state)

        return self._state, self._cov, self._ext

    def correct(self, zs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        n = len(zs)

        z_mean = np.mean(zs, axis=0)
        eps = z_mean - np.dot(np.kron(np.eye(self._dim), self._H), self._state)
        z_center = zs - z_mean
        Z = np.dot(z_center.T, z_center)
        B = lg.cholesky(self._ext / 4 + self._R, lower=True) @ lg.inv(lg.cholesky(self._ext, lower=True))
        B_inv = lg.inv(B)
        S = self._H @ self._single_cov @ self._H.T + lg.det(B)**(2 / self._dim) / n
        S = (S + S.T) / 2
        S_inv = lg.inv(S)
        K = self._single_cov @ self._H.T @ S_inv
        N = S_inv * np.outer(eps, eps)

        # correct inverse wishart parameters
        self._df += n
        self._scale += N + B_inv @ Z @ B_inv.T

        # correct joint state
        self._ext = self._scale / (self._df - 2 * self._dim - 2)
        self._single_cov -= K @ S @ K.T
        self._single_cov = (self._single_cov + self._single_cov.T) / 2
        self._cov = np.kron(self._ext, self._single_cov)
        K_tilde = np.kron(np.eye(self._dim), K)
        self._state += np.dot(K_tilde, eps)

        return self._state, self._cov, self._ext

    def distance(self, zs, **kwargs):
        return super().distance(zs, **kwargs)

    def likelihood(self, zs, **kwargs):
        return super().likelihood(zs, **kwargs)