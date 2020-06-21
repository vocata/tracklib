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
from .base import FilterBase
from tracklib.utils import cholcov


class UKFilterAN(FilterBase):
    '''
    Unscented Kalman filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1) + L_k-1*w_k-1
    z_k = h_k(x_k) + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, f, L, h, M, Q, R, point_generator):
        super().__init__()

        self._f = f
        self._L = L.copy()
        self._h = h
        self._M = M.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._pt_gen = point_generator

    def __str__(self):
        msg = 'Additive noise unscented Kalman filter'
        return msg

    def init(self, state, cov):
        self._state = state.copy()
        self._cov = cov.copy()
        self._pt_gen.init(len(state))
        self._init = True

    def reset(self, state, cov):
        self._state = state.copy()
        self._cov = cov.copy()

    def predict(self, u=None, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'L' in kwargs: self._L[:] = kwargs['L']
            if 'Q' in kwargs: self._Q[:] = kwargs['Q']

        pts_num = self._pt_gen.points_num()
        w_mean, w_cov = self._pt_gen.weights()
        pts = self._pt_gen.sigma_points(self._state, self._cov)

        self.__f_map = []
        self._state = 0
        for pi in range(pts_num):
            tmp = self._f(pts[:, pi], u)
            self.__f_map.append(tmp)
            self._state += w_mean[pi] * tmp

        self._cov = 0
        for pi in range(pts_num):
            err = self.__f_map[pi] - self._state
            self._cov += w_cov[pi] * np.outer(err, err)
        self._cov += self._L @ self._Q @ self._L.T
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov

    def correct(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'M' in kwargs: self._M[:] = kwargs['M']
            if 'R' in kwargs: self._R[:] = kwargs['R']

        pts_num = self._pt_gen.points_num()
        w_mean, w_cov = self._pt_gen.weights()
        pts = self._pt_gen.sigma_points(self._state, self._cov)

        h_map = []
        z_pred = 0
        for pi in range(pts_num):
            tmp = self._h(pts[:, pi])
            h_map.append(tmp)
            z_pred += w_mean[pi] * tmp

        S = 0
        xz_cov = 0
        for pi in range(pts_num):
            z_err = h_map[pi] - z_pred
            S += w_cov[pi] * np.outer(z_err, z_err)
            x_err = self.__f_map[pi] - self._state
            xz_cov += w_cov[pi] * np.outer(x_err, z_err)
        S += self._M @ self._R @ self._M.T
        S = (S + S.T) / 2
        innov = z - z_pred
        K = xz_cov @ lg.inv(S)

        self._state = self._state + K @ innov
        self._cov = self._cov - K @ S @ K.T
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov

    def correct_JPDA(self, zs, probs, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        z_len = len(zs)
        Ms = kwargs['M'] if 'M' in kwargs else [self._M] * z_len
        Rs = kwargs['R'] if 'R' in kwargs else [self._R] * z_len

        pts_num = self._pt_gen.points_num()
        w_mean, w_cov = self._pt_gen.weights()
        pts = self._pt_gen.sigma_points(self._state, self._cov)

        h_map = []
        z_pred = 0
        for pi in range(pts_num):
            tmp = self._h(pts[:, pi])
            h_map.append(tmp)
            z_pred += w_mean[pi] * tmp

        S_base = 0
        xz_cov = 0
        for pi in range(pts_num):
            z_err = h_map[pi] - z_pred
            S_base += w_cov[pi] * np.outer(z_err, z_err)
            x_err = self.__f_map[pi] - self._state
            xz_cov += w_cov[pi] * np.outer(x_err, z_err)

        state_item = 0
        cov_item1 = cov_item2 = 0
        for i in range(z_len):
            S = S_base + Ms[i] @ Rs[i] @ Ms[i].T
            S = (S + S.T) / 2
            K = xz_cov @ lg.inv(S)

            innov = zs[i] - z_pred
            incre = np.dot(K, innov)
            state_item += probs[i] * incre
            cov_item1 += probs[i] * (self._cov - K @ S @ K.T)
            cov_item2 += probs[i] * np.outer(incre, incre)

        self._state = self._state + state_item
        self._cov = (1 - np.sum(probs)) * self._cov + cov_item1 + (cov_item2 - np.outer(state_item, state_item))
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov

    def distance(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        M = kwargs['M'] if 'M' in kwargs else self._M
        R = kwargs['R'] if 'R' in kwargs else self._R

        pts_num = self._pt_gen.points_num()
        w_mean, w_cov = self._pt_gen.weights()
        pts = self._pt_gen.sigma_points(self._state, self._cov)

        h_map = []
        z_pred = 0
        for i in range(pts_num):
            tmp = self._h(pts[:, i])
            h_map.append(tmp)
            z_pred += w_mean[i] * tmp

        S = 0
        for i in range(pts_num):
            z_err = h_map[i] - z_pred
            S += w_cov[i] * np.outer(z_err, z_err)
        S += M @ R @ M.T
        S = (S + S.T) / 2
        innov = z - z_pred
        d = innov @ lg.inv(S) @ innov + np.log(lg.det(S))

        return d

    def likelihood(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        M = kwargs['M'] if 'M' in kwargs else self._M
        R = kwargs['R'] if 'R' in kwargs else self._R

        pts_num = self._pt_gen.points_num()
        w_mean, w_cov = self._pt_gen.weights()
        pts = self._pt_gen.sigma_points(self._state, self._cov)

        h_map = []
        z_pred = 0
        for i in range(pts_num):
            tmp = self._h(pts[:, i])
            h_map.append(tmp)
            z_pred += w_mean[i] * tmp

        S = 0
        for i in range(pts_num):
            z_err = h_map[i] - z_pred
            S += w_cov[i] * np.outer(z_err, z_err)
        S += M @ R @ M.T
        S = (S + S.T) / 2
        innov = z - z_pred
        pdf = 1 / np.sqrt(lg.det(2 * np.pi * S))
        pdf *= np.exp(-innov @ lg.inv(S) @ innov / 2)

        return max(pdf, np.finfo(pdf).tiny)


class UKFilterNAN(FilterBase):
    '''
    Unscented Kalman filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1, w_k)
    z_k = h_k(x_k, v_k)
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, f, h, Q, R, point_generator, epsilon=0.01):
        super().__init__()

        self._f = f
        self._h = h
        # Since the convariance matrix may be semi-definite, adding a small value
        # on the diagonal can make it positive definite.
        self._Q = Q + epsilon * np.diag(Q.diagonal())
        self._R = R + epsilon * np.diag(R.diagonal())
        self._pt_gen = point_generator

    def __str__(self):
        msg = 'Nonadditive noise unscented Kalman filter'
        return msg

    def init(self, state, cov):
        self._state = state.copy()
        self._cov = cov.copy()
        self._pt_gen.init(len(state) + self._Q.shape[0] + self._R.shape[0])
        self._init = True

    def reset(self, state, cov):
        self._state = state.copy()
        self._cov = cov.copy()

    def predict(self, u=None, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        if 'Q' in kwargs: self._Q[:] = kwargs['Q']

        xdim, wdim, vdim = self._state.shape[0], self._Q.shape[0], self._R.shape[0]
        pts_num = self._pt_gen.points_num()
        w_mean, w_cov = self._pt_gen.weights()

        cov_asm = lg.block_diag(self._cov, self._Q, self._R)
        state_asm = np.concatenate((self._state, np.zeros(wdim), np.zeros(vdim)))
        pts_asm = self._pt_gen.sigma_points(state_asm, cov_asm)
        pts = pts_asm[:xdim]
        w_pts = pts_asm[xdim:xdim + wdim]

        self.__f_map = []
        self._state = 0
        for i in range(pts_num):
            tmp = self._f(pts[:, i], u, w_pts[:, i])
            self.__f_map.append(tmp)
            self._state += w_mean[i] * tmp

        self._cov = 0
        for i in range(pts_num):
            err = self.__f_map[i] - self._state
            self._cov += w_cov[i] * np.outer(err, err)
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov

    def correct(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        if 'R' in kwargs: self._R[:] = kwargs['R']

        xdim, wdim, vdim = self._state.shape[0], self._Q.shape[0], self._R.shape[0]
        pts_num = self._pt_gen.points_num()
        w_mean, w_cov = self._pt_gen.weights()

        cov_asm = lg.block_diag(self._cov, self._Q, self._R)
        state_asm = np.concatenate((self._state, np.zeros(wdim), np.zeros(vdim)))
        pts_asm = self._pt_gen.sigma_points(state_asm, cov_asm)
        pts = pts_asm[:xdim]
        v_pts = pts_asm[xdim + wdim:]

        h_map = []
        z_pred = 0
        for i in range(pts_num):
            tmp = self._h(pts[:, i], v_pts[:, i])
            h_map.append(tmp)
            z_pred += w_mean[i] * tmp

        S = 0
        xz_cov = 0
        for i in range(pts_num):
            z_err = h_map[i] - z_pred
            S += w_cov[i] * np.outer(z_err, z_err)
            x_err = self.__f_map[i] - self._state
            xz_cov += w_cov[i] * np.outer(x_err, z_err)
        S = (S + S.T) / 2

        innov = z - z_pred
        K = xz_cov @ lg.inv(S)

        self._state = self._state + K @ innov
        self._cov = self._cov - K @ S @ K.T
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov

    def distance(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        R = kwargs['R'] if 'R' in kwargs else self._R

        pts_num = self._pt_gen.points_num()
        w_mean, w_cov = self._pt_gen.weights()

        cov_asm = lg.block_diag(self._cov, self._Q, R)
        state_asm = np.concatenate((self._state, np.zeros(self._Q.shape[0]), np.zeros(R.shape[0])))
        pts_asm = self._pt_gen.sigma_points(state_asm, cov_asm)
        pts = pts_asm[:len(self._state)]
        v_pts = pts_asm[len(self._state) + self._Q.shape[0]:]

        h_map = []
        z_pred = 0
        for i in range(pts_num):
            tmp = self._h(pts[:, i], v_pts[:, i])
            h_map.append(tmp)
            z_pred += w_mean[i] * tmp

        S = 0
        for i in range(pts_num):
            z_err = h_map[i] - z_pred
            S += w_cov[i] * np.outer(z_err, z_err)
        S = (S + S.T) / 2
        innov = z - z_pred
        d = innov @ lg.inv(S) @ innov + np.log(lg.det(S))

        return d

    def likelihood(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        R = kwargs['R'] if 'R' in kwargs else self._R

        pts_num = self._pt_gen.points_num()
        w_mean, w_cov = self._pt_gen.weights()

        cov_asm = lg.block_diag(self._cov, self._Q, R)
        state_asm = np.concatenate((self._state, np.zeros(self._Q.shape[0]), np.zeros(R.shape[0])))
        pts_asm = self._pt_gen.sigma_points(state_asm, cov_asm)
        pts = pts_asm[:len(self._state)]
        v_pts = pts_asm[len(self._state) + self._Q.shape[0]:]

        h_map = []
        z_pred = 0
        for i in range(pts_num):
            tmp = self._h(pts[:, i], v_pts[:, i])
            h_map.append(tmp)
            z_pred += w_mean[i] * tmp

        S = 0
        for i in range(pts_num):
            z_err = h_map[i] - z_pred
            S += w_cov[i] * np.outer(z_err, z_err)
        S = (S + S.T) / 2
        innov = z - z_pred
        pdf = 1 / np.sqrt(lg.det(2 * np.pi * S))
        pdf *= np.exp(-innov @ lg.inv(S) @ innov / 2)

        return max(pdf, np.finfo(pdf).tiny)


class SimplexSigmaPoints():
    def __init__(self, w0=0):
        assert (0 <= w0 and w0 < 1)
        self._w0 = w0
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
            raise RuntimeError('point generator must be initialized with init() before use')
        return self._dim + 2

    def weights(self):
        if self._init == False:
            raise RuntimeError('point generator must be initialized with init() before use')
        return self._w, self._w

    def sigma_points(self, mean, cov):
        if self._init == False:
            raise RuntimeError('point generator must be initialized with init() before use')
        # P = C * C'
        cov_sqrt = cholcov(cov, lower=True)

        psi = np.zeros(self._dim + 2).tolist()
        psi[0] = np.array([0], dtype=float)
        psi[1] = np.array([-1 / np.sqrt(2 * self._w[1])], dtype=float)
        psi[2] = np.array([1 / np.sqrt(2 * self._w[1])], dtype=float)
        for j in range(2, self._dim + 1):
            for i in range(j + 2):
                if i == 0:
                    psi[i] = np.concatenate((psi[0], np.zeros(1)))
                elif i == j + 1:
                    tmp = np.array([1 / np.sqrt(2 * self._w[j + 1])], dtype=float)
                    psi[i] = np.concatenate((np.zeros(j - 1), tmp))
                else:
                    tmp = np.array([-1 / np.sqrt(2 * self._w[j + 1])], dtype=float)
                    psi[i] = np.concatenate((psi[i], tmp))
        psi = np.array(psi, dtype=float).T
        pts = mean.reshape(-1, 1) + cov_sqrt @ psi
        return pts


class SphericalSimplexSigmaPoints():
    def __init__(self, w0=0):
        assert (0 <= w0 and w0 < 1)
        self._w0 = w0
        self._init = False

    def init(self, dim):
        self._dim = dim
        self._w = np.zeros(dim + 2)
        self._w[0] = self._w0
        self._w[1:] = (1 - self._w0) / (dim + 1)
        self._init = True

    def points_num(self):
        if self._init == False:
            raise RuntimeError('point generator must be initialized with init() before use')
        return self._dim + 2

    def weights(self):
        if self._init == False:
            raise RuntimeError('point generator must be initialized with init() before use')
        return self._w, self._w

    def sigma_points(self, mean, cov):
        if self._init == False:
            raise RuntimeError('point generator must be initialized with init() before use')
        # P = C * C'
        cov_sqrt = cholcov(cov, lower=True)

        psi = np.zeros(self._dim + 2).tolist()
        psi[0] = np.array([0], dtype=float)
        psi[1] = np.array([-1 / np.sqrt(2 * self._w[1])], dtype=float)
        psi[2] = np.array([1 / np.sqrt(2 * self._w[1])], dtype=float)
        for j in range(2, self._dim + 1):
            for i in range(j + 2):
                if i == 0:
                    psi[i] = np.concatenate((psi[0], np.zeros(1)))
                elif i == j + 1:
                    tmp = np.array([j / np.sqrt(j * (j + 1) * self._w[1])], dtype=float)
                    psi[i] = np.concatenate((np.zeros(j - 1), tmp))
                else:
                    tmp = np.array([-1 / np.sqrt(j * (j + 1) * self._w[1])], dtype=float)
                    psi[i] = np.concatenate((psi[i], tmp))
        psi = np.array(psi, dtype=float).T
        pts = mean.reshape(-1, 1) + cov_sqrt @ psi
        return pts


class SymmetricSigmaPoints():
    '''
    Note that if symmetrical sigma points are selected, UKF is CKF.
    This symmetric sample point set often results in better statistical
    stability and avoids divergence which might occur in UKF, especially
    when running in a single-precision platform. 
    '''
    def __init__(self):
        self._init = False

    def init(self, dim):
        self._dim = dim
        self._w = np.full(2 * dim, 1 / (2 * dim), dtype=float)
        self._init = True

    def points_num(self):
        if self._init == False:
            raise RuntimeError('point generator must be initialized with init() before use')
        return 2 * self._dim

    def weights(self):
        if self._init == False:
            raise RuntimeError('point generator must be initialized with init() before use')
        return self._w, self._w

    def sigma_points(self, mean, cov):
        if self._init == False:
            raise RuntimeError('point generator must be initialized with init() before use')
        # P = C * C'
        cov_sqrt = cholcov(cov, lower=True)

        pts = np.zeros((self._dim, 2 * self._dim))
        for i in range(self._dim):
            pts[:, i] = mean + np.sqrt(self._dim) * cov_sqrt[:, i]
            pts[:, self._dim + i] = mean - np.sqrt(self._dim) * cov_sqrt[:, i]
        return pts


class ScaledSigmaPoints():
    def __init__(self, alpha=1, beta=2, kappa=None):
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
            the scaling parameter that is usually set to 0. Smaller values correspond
            to sigma points closer to the mean state. The spread is proportional to the
            square-root of kappa. if kappa = 3 - n, n is the dimension of state, it is
            possible to match some of the fourth order terms when state is Gaussian.
        Note that a CKF is essentially equivalent to a UKF when the parameters are set 
        to alpha = 1, beta = 0, and kappa = 0
        '''
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._init = False

    def init(self, dim):
        self._dim = dim

        if self._kappa is None:
            self._kappa = 3 - dim
        self._lamb = self._alpha**2 * (dim + self._kappa) - dim
        self._w_mean = np.full(2 * dim + 1, 1 / (2 * (dim + self._lamb)), dtype=float)
        self._w_mean[-1] = self._lamb / (dim + self._lamb)
        self._w_cov = self._w_mean.copy()
        self._w_cov[-1] = self._w_mean[-1] + (1 - self._alpha**2 + self._beta)
        self._init = True

    def points_num(self):
        if self._init == False:
            raise RuntimeError('point generator must be initialized with init() before use')
        return 2 * self._dim + 1

    def weights(self):
        if self._init == False:
            raise RuntimeError('point generator must be initialized with init() before use')
        return self._w_mean, self._w_cov

    def sigma_points(self, mean, cov):
        if self._init == False:
            raise RuntimeError('point generator must be initialized with init() before use')
        # P = C * C'
        cov_sqrt = cholcov(cov, lower=True)

        pts = np.zeros((self._dim, 2 * self._dim + 1))
        pts[:, -1] = mean
        for i in range(self._dim):
            pts[:, i] = mean + np.sqrt(self._dim + self._lamb) * cov_sqrt[:, i]
            pts[:, self._dim + i] = mean - np.sqrt(self._dim + self._lamb) * cov_sqrt[:, i]
        return pts
