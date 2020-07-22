# -*- coding: utf-8 -*-
'''
Extended object tracker

REFERENCE:
[1]. 
'''
from __future__ import division, absolute_import, print_function


__all__ = ['KochEOFilter', 'FeldmannEOFilter']

import numpy as np
import scipy.linalg as lg
from .base import EOFilterBase


class KochEOFilter(EOFilterBase):
    '''
    Extended object particle filter
    '''
    def __init__(self, F, H, Q, interval, tau, dim=2):
        self._F = F.copy()
        self._H = H.copy()
        self._Q = Q.copy()
        self._at = np.exp(-interval / tau)      # attenuation factor
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

        self._single_cov = self._F @ self._single_cov @ self._F + self._Q

        # predict inverse wishart parameters
        df = self._df
        self._df = self._at * self._df
        w = (self._df - 2 * self._dim - 2) / (df - 2 * self._dim - 2)
        self._scale = w * self._scale

        # predict joint state
        self._ext = self._scale / (self._df - 2 * self._dim - 2)
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
        S = self._H @ self._single_cov @ self._H.T + 1 / n
        S_inv = lg.inv(S)
        K = self._single_cov @ self._H.T @ S_inv
        N = S_inv * np.outer(eps, eps)

        # correct inverse wishart parameters
        self._df += n
        self._scale += N + Z

        # correct joint state
        self._ext = self._scale / (self._df - 2 * self._dim - 2)
        self._single_cov -= K @ S @ K.T
        self._cov = np.kron(self._ext, self._single_cov)
        K_tilde = np.kron(np.eye(self._dim), K)
        self._state += np.dot(K_tilde, eps)

        return self._state, self._cov, self._ext

    def distance(self, z, **kwargs):
        return super().distance(z, **kwargs)

    def likelihood(self, z, **kwargs):
        return super().likelihood(z, **kwargs)


class FeldmannEOFilter(EOFilterBase):
    '''
    Extended object particle filter
    '''
    def __init__(self, F, H, Q, R, interval, tau, dim=2):
        self._F = F.copy()
        self._H = H.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._at = np.exp(-interval / tau)      # attenuation
        self._dim = dim

    def init(self, state, cov, df, extension):
        self._df = df
        self._scale = extension * (df - 2 * self._dim - 2)

        self._state = state.copy()
        self._cov = cov.copy()
        self._ext = extension.copy()
        self._init = True

    def predict(self):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        # predict inverse wishart parameters
        df = self._df
        self._df = 2 * self._dim + 2 + self._at * (self._df - 2 * self._dim - 2)
        w = (self._df - 2 * self._dim - 2) / (df - 2 * self._dim - 2)
        self._scale = w * self._scale

        # predict joint state
        self._ext = self._scale / (self._df - 2 * self._dim - 2)
        self._cov = self._F @ self._cov @ self._F + np.kron(self._ext, self._Q)
        self._state = np.dot(self._F, self._state)
        return self._state, self._cov, self._ext

    def correct(self, zs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        n = len(zs)

        z_mean = np.mean(zs, axis=0)
        eps = z_mean - np.dot(self._H, self._state)
        z_center = zs - z_mean
        Z = np.dot(z_center.T, z_center)

        X_hat = self._scale / (self._df - 2 * self._dim - 2)
        Y = X_hat / 4 + self._R
        S = self._H @ self._cov @ self._H.T + Y / n
        S_inv = lg.inv(S)
        K = self._cov @ self._H.T @ S_inv
        X_chol = lg.cholesky(X_hat, lower=True)
        S_chol = lg.cholesky(S_inv, lower=True)
        Y_chol = lg.cholesky(lg.inv(Y), lower=True)
        N = np.outer(eps, eps)
        N_hat = X_chol @ S_chol @ N @ S_chol.T @ X_chol.T
        Z_hat = X_chol @ Y_chol @ Z @ Y_chol.T @ X_chol.T

        # correct inverse wishart parameters
        self._df += n
        self._scale += N_hat + Z_hat

        # correct joint state
        self._ext = self._scale / (self._df - 2 * self._dim - 2)
        self._cov -= K @ S @ K.T
        self._state += np.dot(K, eps)

        return self._state, self._cov, self._ext

    def distance(self, z, **kwargs):
        return super().distance(z, **kwargs)

    def likelihood(self, z, **kwargs):
        return super().likelihood(z, **kwargs)
