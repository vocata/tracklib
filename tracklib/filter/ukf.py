# -*- coding: utf-8 -*-
'''
Unscented Kalman filter

REFERENCE:
[1]. D. Simon, "Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches," John Wiley and Sons, Inc., 2006.
[2]. S. Julier and J. Uhlmann, "Unscented filtering and nonlinear estimation," Proceedings of the IEEE, 92(3), pp. 401-422 (March 2004)
[3]. M. GREWAL and A. ANDREWS, "Kalman Filtering Theory and Practice Using MATLAB," John Wiley & Sons, Inc., Hoboken, New Jersey, 2015.
[4]. S. Haykin, "Kalman Filtering and Neural Networks," John Wiley & Sons, New York, 2001.
[5]. S. Julier, “The spherical simplex unscented transformation,” American Con- trol Conference, pp. 2430-2434, 2003.
[6]. E. A. Wan and R. Van Der Merwe, "The unscented Kalman filter for nonlinear estimation," Proceedings of the IEEE, 2000, pp. 153-158.
[7]. I. Arasaratnam and S. Haykin, "Cubature Kalman Filters," in IEEE Transactions on Automatic Control, vol. 54, no. 6, pp. 1254-1269, June 2009.
'''
from __future__ import division, absolute_import, print_function


__all__ = [
    'UKFilterAN', 'UKFilterNAN', 'SimplexSigmaPoints',
    'SphericalSimplexSigmaPoints', 'SymmetricSigmaPoints', 'ScaledSigmaPoints'
]

import numpy as np
import scipy.linalg as lg
from .base import KFBase


class UKFilterAN(KFBase):
    '''
    Unscented Kalman filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1) + L_k-1*w_k-1
    z_k = h_k(x_k) + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, f, L, h, M, Q, R, factory):
        super().__init__()

        self._f = f
        self._L = L
        self._h = h
        self._M = M
        self._Q = Q
        self._R = R
        self._factory = factory

    def __str__(self):
        msg = 'Additive noise unscented Kalman filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        self._prior_state = state
        self._prior_cov = cov
        self._post_state = state
        self._post_cov = cov
        self._factory.init(len(state))
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

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
        self._prior_cov += self._L @ self._Q @ self._L.T
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

        self._stage = 1

    def update(self, z, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

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
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2

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
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict(u, **kw)
        self.update(z, **kw)


class UKFilterNAN(KFBase):
    '''
    Unscented Kalman filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1, w_k)
    z_k = h_k(x_k, v_k)
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, f, h, Q, R, factory, epsilon=0.01):
        super().__init__()

        self._f = f
        self._h = h
        # Since the convariance matrix may be semi-definite, adding a small value
        # on the diagonal can make it positive definite.
        self._Q = Q + epsilon * np.diag(Q.diagonal())
        self._R = R + epsilon * np.diag(R.diagonal())
        self._factory = factory

    def __str__(self):
        msg = 'Nonadditive noise unscented Kalman filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        self._prior_state = state
        self._prior_cov = cov
        self._post_state = state
        self._post_cov = cov
        self._factory.init(len(state) + self._Q.shape[0] + self._R.shape[0])
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        x_dim = len(self._prior_state)
        w_dim = self._Q.shape[0]
        v_dim = self._R.shape[0]

        if len(kw) > 0:
            if 'f' in kw: self._f = kw['f']
            if 'Q' in kw: self._Q = kw['Q']

        pts_num = self._factory.points_num()
        w_mean, w_cov = self._factory.weights()
        post_cov_asm = lg.block_diag(self._post_cov, self._Q, self._R)
        post_state_asm = np.concatenate((self._post_state, np.zeros(w_dim), np.zeros(v_dim)))
        pts_asm = self._factory.sigma_points(post_state_asm, post_cov_asm)
        pts = pts_asm[:x_dim, :]
        w_pts = pts_asm[x_dim:x_dim + w_dim, :]

        self.__f_map = np.zeros_like(pts)
        self._prior_state = 0
        for i in range(pts_num):
            f_map = self._f(pts[:, i], u, w_pts[:, i])
            self.__f_map[:, i] = f_map
            self._prior_state += w_mean[i] * f_map

        self._prior_cov = 0
        for i in range(pts_num):
            err = self.__f_map[:, i] - self._prior_state
            self._prior_cov += w_cov[i] * np.outer(err, err)
        self._prior_cov = (self._prior_cov + self._prior_cov.T) / 2

        self._stage = 1

    def update(self, z, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        x_dim = len(self._prior_state)
        w_dim = self._Q.shape[0]
        v_dim = self._R.shape[0]

        if len(kw) > 0:
            if 'h' in kw: self._h = kw['h']
            if 'R' in kw: self._R = kw['R']

        z_dim = len(z)
        pts_num = self._factory.points_num()
        w_mean, w_cov = self._factory.weights()

        prior_cov_asm = lg.block_diag(self._prior_cov, self._Q, self._R)
        prior_state_asm = np.concatenate((self._prior_state, np.zeros(w_dim), np.zeros(v_dim)))
        pts_asm = self._factory.sigma_points(prior_state_asm, prior_cov_asm)
        pts = pts_asm[:x_dim, :]
        v_pts = pts_asm[x_dim + w_dim:, :]

        self.__h_map = np.zeros((z_dim, pts_num))
        z_prior = 0
        for i in range(pts_num):
            h_map = self._h(pts[:, i], v_pts[:, i])
            self.__h_map[:, i] = h_map
            z_prior += w_mean[i] * h_map

        self._innov_cov = 0
        xz_cov = 0
        for i in range(pts_num):
            z_err = self.__h_map[:, i] - z_prior
            self._innov_cov += w_cov[i] * np.outer(z_err, z_err)
            x_err = self.__f_map[:, i] - self._prior_state
            xz_cov += w_cov[i] * np.outer(x_err, z_err)
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2

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
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict(u, **kw)
        self.update(z, **kw)


class SimplexSigmaPoints():
    def __init__(self, w0=0, decompose='cholesky'):
        assert (0 <= w0 and w0 < 1)
        self._w0 = w0
        self._decompose = decompose
        self._init = False

    def init(self, dim):
        self._dim = dim
        self._w = np.zeros(dim + 2)
        self._w[0] = self._w0
        self._w[1: 3] = (1 - self._w0) / 2**dim
        self._w[3: dim + 2] = (2**np.arange(1, dim)) * self._w[1]
        self._init = True

    def points_num(self):
        if self._init == False:
            raise RuntimeError('the factory must be initialized with init() before use')
        return self._dim + 2

    def weights(self):
        if self._init == False:
            raise RuntimeError('the factory must be initialized with init() before use')
        return self._w, self._w

    def sigma_points(self, mean, cov):
        if self._init == False:
            raise RuntimeError('the factory must be initialized with init() before use')
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
    def __init__(self, w0=0, decompose='cholesky'):
        assert (0 <= w0 and w0 < 1)
        self._w0 = w0
        self._decompose = decompose
        self._init = False

    def init(self, dim):
        self._dim = dim
        self._w = np.zeros(dim + 2)
        self._w[0] = self._w0
        self._w[1:] = (1 - self._w0) / (dim + 1)
        self._init = True

    def points_num(self):
        if self._init == False:
            raise RuntimeError('the factory must be initialized with init() before use')
        return self._dim + 2

    def weights(self):
        if self._init == False:
            raise RuntimeError('the factory must be initialized with init() before use')
        return self._w, self._w

    def sigma_points(self, mean, cov):
        if self._init == False:
            raise RuntimeError('the factory must be initialized with init() before use')
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


class SymmetricSigmaPoints():
    '''
    Note that if select symmetric sigma points then UKF will become CKF
    '''
    def __init__(self, decompose='cholesky'):
        self._decompose = decompose
        self._init = False

    def init(self, dim):
        self._dim = dim
        self._w = np.ones(2 * dim) / (2 * dim)
        self._init = True

    def points_num(self):
        if self._init == False:
            raise RuntimeError('the factory must be initialized with init() before use')
        return 2 * self._dim

    def weights(self):
        if self._init == False:
            raise RuntimeError('the factory must be initialized with init() before use')
        return self._w, self._w

    def sigma_points(self, mean, cov):
        if self._init == False:
            raise RuntimeError('the factory must be initialized with init() before use')
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
    def __init__(self, alpha=1, beta=2, kappa=None, decompose='cholesky'):
        '''
        alpha:
            Determines the spread of the sigma points around the mean state value.
            It is usually a small positive value. The spread of sigma points is
            proportional to alpha. Smaller values correspond to sigma points closer
            to the mean state.
        beta
            Incorporates prior knowledge of the distribution of the state. For Gaussian
            distributions, beta = 2 is optimal.
        kappa
            A second scaling parameter that is usually set to 0. Smaller values correspond
            to sigma points closer to the mean state. The spread is proportional to the
            square-root of kappa. if kappa = 3 - n, n is the dimension of state, it is
            possible to match some of the fourth order terms when state is Gaussian.
        '''
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._decompose = decompose
        self._init = False

    def init(self, dim):
        self._dim = dim

        if self._kappa is None:
            self._kappa = 3 - dim
        self._lamb = self._alpha**2 * (dim + self._kappa) - dim
        self._w_mean = np.ones(2 * dim + 1) / (2 * (dim + self._lamb))
        self._w_mean[-1] = self._lamb / (dim + self._lamb)
        self._w_cov = self._w_mean.copy()
        self._w_cov[-1] = self._w_mean[-1] + (1 - self._alpha**2 + self._beta)
        self._init = True

    def points_num(self):
        if self._init == False:
            raise RuntimeError('the factory must be initialized with init() before use')
        return 2 * self._dim + 1

    def weights(self):
        if self._init == False:
            raise RuntimeError('the factory must be initialized with init() before use')
        return self._w_mean, self._w_cov

    def sigma_points(self, mean, cov):
        if self._init == False:
            raise RuntimeError('the factory must be initialized with init() before use')
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
            pts[:, self._dim + i] = mean - np.sqrt(self._dim + self._lamb) * cov_sqrt[:, i]
        return pts
