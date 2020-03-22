# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
from .kfbase import KFBase

__all__ = [
    'UKFilter', 'SimplexSigmaPoints', 'SphericalSimplexSigmaPoints',
    'SymmetricSigmaPoint', 'ScaledSigmaPoints'
]


class UKFilter(KFBase):
    '''
    Unscented Kalman filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1) + L_k-1*w_k-1
    z_k = h_k(x_k) + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, f, L, h, M, Q, R, factory, at=1):
        super().__init__()

        self._f = f
        self._L = L
        self._h = h
        self._M = M
        self._Q = Q
        self._R = R
        self._factory = factory
        self._at = at

    def __str__(self):
        msg = 'Unscented Kalman filter'
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

        if len(kw) > 0:
            if 'f' in kw: self._f = kw['f']
            if 'L' in kw: self._L = kw['L']
            if 'Q' in kw: self._Q = kw['Q']

        pts_num = self._factory.points_num()
        w_mean, w_cov = self._factory.weights()
        points = self._factory.sigma_points(self._post_state, self._post_cov)

        self.__f_map = np.zeros_like(points)
        self._prior_state = 0
        for i in range(pts_num):
            f_map = self._f(points[:, i], u)
            self.__f_map[:, i] = f_map
            self._prior_state += w_mean[i] * f_map

        self._prior_cov = 0
        for i in range(pts_num):
            err = self.__f_map[:, i] - self._prior_state
            self._prior_cov += w_cov[i] * np.outer(err, err)
        self._prior_cov = self._at**2 * self._prior_cov + self._L @ self._Q @ self._L.T
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

        self._stage = 1

    def update(self, z, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        if len(kw) > 0:
            if 'h' in kw: self._h = kw['h']
            if 'M' in kw: self._M = kw['M']
            if 'R' in kw: self._R = kw['R']

        z_dim = len(z)
        pts_num = self._factory.points_num()
        w_mean, w_cov = self._factory.weights()
        points = self._factory.sigma_points(self._prior_state, self._prior_cov)

        self.__h_map = np.zeros((z_dim, pts_num))
        z_prior = 0
        for i in range(pts_num):
            h_map = self._h(points[:, i])
            self.__h_map[:, i] = h_map
            z_prior += w_mean[i] * h_map

        self._innov_cov = 0
        xz_cov = 0
        for i in range(pts_num):
            z_err = self.__h_map[:, i] - z_prior
            self._innov_cov += w_cov[i] * np.outer(z_err, z_err)
            x_err = self.__f_map[:, i] - self._prior_state
            xz_cov += w_cov[i] * np.outer(x_err, z_err)
        self._innov_cov += self._M @ self._R @ self._M.T
        self._innov_cov = (self._innov_cov + self._innov_cov) / 2

        self._innov = z - z_prior
        self._gain = xz_cov @ lg.inv(self._innov_cov)
        self._post_state = self._prior_state + self._gain @ self._innov
        self._post_cov = self._prior_cov - self._gain @ self._innov_cov @ self._gain.T
        self._post_cov = (self._post_cov + self._post_cov.T) / 2

        self._len += 1
        self._stage = 0  # update finished

    def step(self, z, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        self.predict(u, **kw)
        self.update(z, **kw)


class SimplexSigmaPoints():
    def __init__(self, dim, w0=0, decompose='cholesky'):
        assert (0 <= w0 and w0 < 1)
        self._dim = dim
        self._decompose = decompose

        self._w = np.zeros(dim + 2)
        self._w[0] = w0
        self._w[1: 3] = (1 - w0) / 2**dim
        self._w[3: dim + 2] = (2**np.arange(1, dim)) * self._w[1]

    def points_num(self):
        return self._dim + 2

    def weights(self):
        return self._w, self._w

    def sigma_points(self, mean, cov):
        # P = C * C'
        if self._decompose.lower() == 'cholesky':
            cov_sqrt = lg.cholesky(cov, lower=True)
        elif self._decompose.lower() == 'svd':
            U, s, V = lg.svd(cov)
            cov_sqrt = U @ np.diag(np.sqrt(s)) @ V.T
        else:
            raise ValueError('unknown decomposition: %s' % self._decompose)

        psi = np.zeros(self._dim + 2).tolist()
        psi[0] = np.array([0])
        psi[1] = np.array([-1 / np.sqrt(2 * self._w[1])])
        psi[2] = np.array([1 / np.sqrt(2 * self._w[1])])
        for j in range(2, self._dim + 1):
            for i in range(j + 2):
                if i == 0:
                    psi[i] = np.concatenate((psi[0], np.zeros(1)))
                elif i == j + 1:
                    tmp = np.array([1 / np.sqrt(2 * self._w[j + 1])])
                    psi[i] = np.concatenate((np.zeros(j - 1), tmp))
                else:
                    tmp = np.array([-1 / np.sqrt(2 * self._w[j + 1])])
                    psi[i] = np.concatenate((psi[i], tmp))
        psi = np.array(psi).T
        pts = mean.reshape(-1, 1) + cov_sqrt @ psi
        return pts


class SphericalSimplexSigmaPoints():
    def __init__(self, dim, w0=0, decompose='cholesky'):
        assert (0 <= w0 and w0 < 1)
        self._dim = dim
        self._decompose = decompose

        self._w = np.zeros(dim + 2)
        self._w[0] = w0
        self._w[1:] = (1 - w0) / (dim + 1)

    def points_num(self):
        return self._dim + 2

    def weights(self):
        return self._w, self._w

    def sigma_points(self, mean, cov):
        # P = C * C'
        if self._decompose.lower() == 'cholesky':
            cov_sqrt = lg.cholesky(cov, lower=True)
        elif self._decompose.lower() == 'svd':
            U, s, V = lg.svd(cov)
            cov_sqrt = U @ np.diag(np.sqrt(s)) @ V.T
        else:
            raise ValueError('unknown decomposition: %s' % self._decompose)

        psi = np.zeros(self._dim + 2).tolist()
        psi[0] = np.array([0])
        psi[1] = np.array([-1 / np.sqrt(2 * self._w[1])])
        psi[2] = np.array([1 / np.sqrt(2 * self._w[1])])
        for j in range(2, self._dim + 1):
            for i in range(j + 2):
                if i == 0:
                    psi[i] = np.concatenate((psi[0], np.zeros(1)))
                elif i == j + 1:
                    tmp = np.array([j / np.sqrt(j * (j + 1) * self._w[1])])
                    psi[i] = np.concatenate((np.zeros(j - 1), tmp))
                else:
                    tmp = np.array([-1 / np.sqrt(j * (j + 1) * self._w[1])])
                    psi[i] = np.concatenate((psi[i], tmp))
        psi = np.array(psi).T
        pts = mean.reshape(-1, 1) + cov_sqrt @ psi
        return pts


class SymmetricSigmaPoint():
    def __init__(self, dim, decompose='cholesky'):
        self._dim = dim
        self._decompose = decompose

        self._w = np.ones(2 * dim) / (2 * dim)

    def points_num(self):
        return 2 * self._dim

    def weights(self):
        return self._w, self._w

    def sigma_points(self, mean, cov):
        # P = C * C'
        if self._decompose.lower() == 'cholesky':
            cov_sqrt = lg.cholesky(cov, lower=True)
        elif self._decompose.lower() == 'svd':
            U, s, V = lg.svd(cov)
            cov_sqrt = U @ np.diag(np.sqrt(s)) @ V.T
        else:
            raise ValueError('unknown decomposition: %s' % self._decompose)

        pts = np.zeros((self._dim, 2 * self._dim))
        for i in range(self._dim):
            pts[:, i] = mean + np.sqrt(self._dim) * cov_sqrt[:, i]
            pts[:, self._dim + i] = mean - np.sqrt(self._dim) * cov_sqrt[:, i]
        return pts


class ScaledSigmaPoints():
    def __init__(self, dim, kappa, alpha=1, beta=2, decompose='svd'):
        self._dim = dim
        self._decompose = decompose

        self._lamb = alpha**2 * (dim + kappa) - dim
        self._w_mean = np.ones(2 * dim + 1) / (2 * (dim + self._lamb))
        self._w_mean[-1] = self._lamb / (dim + self._lamb)
        self._w_cov = self._w_mean.copy()
        self._w_cov[-1] = self._w_mean[-1] + (1 - alpha**2 + beta)

    def points_num(self):
        return 2 * self._dim + 1

    def weights(self):
        return self._w_mean, self._w_cov

    def sigma_points(self, mean, cov):
        # P = C * C'
        if self._decompose == 'cholesky':
            cov_sqrt = lg.cholesky(cov, lower=True)
        elif self._decompose == 'svd':
            U, s, V = lg.svd(cov)
            cov_sqrt = U @ np.diag(np.sqrt(s)) @ V.T
        else:
            raise ValueError('unknown decomposition: %s' % self._decompose)

        pts = np.zeros((self._dim, 2 * self._dim + 1))
        pts[:, -1] = mean
        for i in range(self._dim):
            pts[:, i] = mean + np.sqrt(self._dim + self._lamb) * cov_sqrt[:, i]
            pts[:, self._dim +
                i] = mean - np.sqrt(self._dim + self._lamb) * cov_sqrt[:, i]
        return pts
