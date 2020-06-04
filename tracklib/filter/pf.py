# -*- coding: utf-8 -*-
'''
Particle filter

REFERENCE:
[1]. D. Simon, "Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches," John Wiley and Sons, Inc., 2006.
[2]. M. S. Arulampalam, S. Maskell, N. Gordon and T. Clapp, "A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking," in IEEE Transactions on Signal Processing, vol. 50, no. 2, pp. 174-188, Feb. 2002.
[3]. A. Doucet, J. F. G. de Freitas, and N. J. Gordon, Eds. "Sequential Monte Carlo Methods in Practice," New York: Springer-Verlag, 2001.
[4]. A. Doucet, S. Godsill and C. Andrieu, "On sequential Monte Carlo sampling methods for Bayesian filtering," Statistics and Computing 10, 197–208 (2000).
[5]. J. Candy, "Bayesian signal processing: Classical, modern, and particle filtering methods, second edition", Wiley Online Books, 2016
'''
from __future__ import division, absolute_import, print_function


__all__ = ['SIRPFilter', 'RPFilter', 'EpanechnikovKernal', 'GaussianKernal']

import numpy as np
import scipy.linalg as lg
from .base import FilterBase
from tracklib.utils import multi_normal, disc_random, cholcov


class SIRPFilter(FilterBase):
    '''
    Sampling importance resampling (SIR) filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1) + L_k-1*w_k-1
    z_k = h_k(x_k) + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other

    note that the transition density is selected as its proposal distribution in SIR filter,
    which is also called condensation filter.
    '''
    def __init__(self, f, L, h, M, Q, R, Ns, Neff, resample_alg='roulette'):
        super().__init__()

        self._f = f
        self._L = L.copy()
        self._h = h
        self._M = M.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._Ns = Ns
        self._Neff = Neff
        self._resample_alg = resample_alg

    def __str__(self):
        msg = 'SIR particle filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        self._state = state.copy()
        self._cov = state.copy()
        self._samples = multi_normal(state, cov, self._Ns, axis=0)
        self._weights = np.full(self._Ns, 1 / self._Ns, dtype=float)
        self._init = True

    def reset(self, state, cov):
        self._state = state.copy()
        self._cov = state.copy()
        self._samples = multi_normal(state, cov, self._Ns, axis=0)
        self._weights = np.full(self._Ns, 1 / self._Ns, dtype=float)

    def predict(self, u=None, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'L' in kwargs: self._L[:] = kwargs['L']
            if 'Q' in kwargs: self._Q[:] = kwargs['Q']

        # compute prior state and covariance
        # E[f(x_k)+w_k|z_1:k] = E[f(x_k)|z_1:k] = Σf(x_k^i)*w^i
        f_map = [self._f(self._samples[i], u) for i in range(self._Ns)]
        self._state = np.dot(self._weights, f_map)
        self._cov = 0
        for i in range(self._Ns):
            err = f_map[i] - self._state
            self._cov += self._weights[i] * np.outer(err, err)
        self._cov = (self._cov + self._cov.T) / 2

        # update samples
        Q_tilde = self._L @ self._Q @ self._L.T
        proc_noi = multi_normal(0, Q_tilde, self._Ns, axis=0)
        for i in range(self._Ns):
            self._samples[i] = f_map[i] + proc_noi[i]

        return self._state, self._cov

    def correct(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'M' in kwargs: self._M[:] = kwargs['M']
            if 'R' in kwargs: self._R[:] = kwargs['R']

        # update weights
        R_tilde = self._M @ self._R @ self._M.T
        for i in range(self._Ns):
            noi = z - self._h(self._samples[i])
            pdf = 1 / np.sqrt(lg.det(2 * np.pi * R_tilde))
            pdf *= np.exp(-noi @ lg.inv(R_tilde) @ noi / 2)
            self._weights[i] *= max(pdf, np.finfo(pdf).tiny)
        self._weights /= np.sum(self._weights)

        # resample
        Neff = 1 / np.sum(self._weights**2)
        if Neff <= self._Neff:
            self._samples[:], _ = disc_random(self._weights, self._Ns, self._samples, alg=self._resample_alg)
            self._weights[:] = 1 / self._Ns

        # compute post state and covariance
        self._state = np.dot(self._weights, self._samples)
        self._cov = 0
        for i in range(self._Ns):
            err = self._samples[i] - self._state
            self._cov += self._weights[i] * np.outer(err, err)
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov

    def distance(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        M = kwargs['M'] if 'M' in kwargs else self._M
        R = kwargs['R'] if 'R' in kwargs else self._R

        R_tilde = M @ R @ M.T
        h_map = [self._h(self._samples[i]) for i in range(self._Ns)]
        z_pred = np.dot(self._weights, h_map)
        innov = z - z_pred
        S = 0
        for i in range(self._Ns):
            err = h_map[i] - z_pred
            S += self._weights[i] * np.outer(err, err)
        S += R_tilde
        S = (S + S.T) / 2
        d = innov @ lg.inv(S) @ innov + np.log(lg.det(S))

        return d

    def likelihood(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        M = kwargs['M'] if 'M' in kwargs else self._M
        R = kwargs['R'] if 'R' in kwargs else self._R

        R_tilde = M @ R @ M.T
        h_map = [self._h(self._samples[i]) for i in range(self._Ns)]
        z_pred = np.dot(self._weights, h_map)
        innov = z - z_pred
        S = 0
        for i in range(self._Ns):
            err = h_map[i] - z_pred
            S += self._weights[i] * np.outer(err, err)
        S += R_tilde
        S = (S + S.T) / 2
        pdf = 1 / np.sqrt(lg.det(2 * np.pi * S))
        pdf *= np.exp(-innov @ lg.inv(S) @ innov / 2)

        return max(pdf, np.finfo(pdf).tiny)

class RPFilter(FilterBase):
    '''
    Regularized particle filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1) + L_k-1*w_k-1
    z_k = h_k(x_k) + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    mainly solves sample impoverishment problem
    '''
    def __init__(self, f, L, h, M, Q, R, Ns, Neff, kernal, resample_alg='roulette'):
        super().__init__()

        self._f = f
        self._L = L.copy()
        self._h = h
        self._M = M.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._Ns = Ns
        self._Neff = Neff
        self._kernal = kernal
        self._resample_alg = resample_alg

    def __str__(self):
        msg = 'Regularized particle filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        self._state = state.copy()
        self._cov = cov.copy()
        self._samples = multi_normal(state, cov, Ns=self._Ns, axis=0)
        self._weights = np.full(self._Ns, 1 / self._Ns, dtype=float)
        self._init = True

    def reset(self, state, cov):
        self._state = state.copy()
        self._cov = cov.copy()
        self._samples = multi_normal(state, cov, Ns=self._Ns, axis=0)
        self._weights = np.full(self._Ns, 1 / self._Ns, dtype=float)

    def predict(self, u=None, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'L' in kwargs: self._L[:] = kwargs['L']
            if 'Q' in kwargs: self._Q[:] = kwargs['Q']

        # compute prior state and covariance
        # E[f(x_k)+w_k|z_1:k] = E[f(x_k)|z_1:k] = Σf(x_k^i)*w^i
        f_map = [self._f(self._samples[i], u) for i in range(self._Ns)]
        self._state = np.dot(self._weights, f_map)
        self._cov = 0
        for i in range(self._Ns):
            err = f_map[i] - self._state
            self._cov += self._weights[i] * np.outer(err, err)
        self._cov = (self._cov + self._cov.T) / 2

        # update samples
        Q_tilde = self._L @ self._Q @ self._L.T
        proc_noi = multi_normal(0, Q_tilde, self._Ns, axis=0)
        for i in range(self._Ns):
            self._samples[i] = f_map[i] + proc_noi[i]

        return self._state, self._cov

    def correct(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        if len(kwargs) > 0:
            if 'M' in kwargs: self._M[:] = kwargs['M']
            if 'R' in kwargs: self._R[:] = kwargs['R']

        # update weights
        R_tilde = self._M @ self._R @ self._M.T
        for i in range(self._Ns):
            noi = z - self._h(self._samples[i])
            pdf = 1 / np.sqrt(lg.det(2 * np.pi * R_tilde))
            pdf *= np.exp(-noi @ lg.inv(R_tilde) @ noi / 2)
            self._weights[i] *= max(pdf, np.finfo(pdf).tiny)
        self._weights /= np.sum(self._weights)

        # resample and regularize
        Neff = 1 / np.sum(self._weights**2)
        if Neff <= self._Neff:
            self._samples[:], self._weights[:] = self._kernal.resample(
                self._samples, self._weights, resample_alg=self._resample_alg)

        # compute post state and covariance
        self._state = np.dot(self._weights, self._samples)
        self._cov = 0
        for i in range(self._Ns):
            err = self._samples[i] - self._state
            self._cov += self._weights[i] * np.outer(err, err)
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov

    def distance(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        M = kwargs['M'] if 'M' in kwargs else self._M
        R = kwargs['R'] if 'R' in kwargs else self._R

        R_tilde = M @ R @ M.T
        h_map = [self._h(self._samples[i]) for i in range(self._Ns)]
        z_pred = np.dot(self._weights, h_map)
        innov = z - z_pred
        S = 0
        for i in range(self._Ns):
            err = h_map[i] - z_pred
            S += self._weights[i] * np.outer(err, err)
        S += R_tilde
        S = (S + S.T) / 2
        d = innov @ lg.inv(S) @ innov + np.log(lg.det(S))

        return d

    def likelihood(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        M = kwargs['M'] if 'M' in kwargs else self._M
        R = kwargs['R'] if 'R' in kwargs else self._R

        R_tilde = M @ R @ M.T
        h_map = [self._h(self._samples[i]) for i in range(self._Ns)]
        z_pred = np.dot(self._weights, h_map)
        innov = z - z_pred
        S = 0
        for i in range(self._Ns):
            err = h_map[i] - z_pred
            S += self._weights[i] * np.outer(err, err)
        S += R_tilde
        S = (S + S.T) / 2
        pdf = 1 / np.sqrt(lg.det(2 * np.pi * S))
        pdf *= np.exp(-innov @ lg.inv(S) @ innov / 2)

        return max(pdf, np.finfo(pdf).tiny)


class EpanechnikovKernal():
    def __init__(self, dim, Ns):
        vol = EpanechnikovKernal.unit_hypershpere_volumn(dim)
        n = dim + 4
        self.opt_bandwidth = ((8 * n * (2 * np.sqrt(np.pi))**dim / vol) / Ns)**(1 / n)
        self._dim = dim
        self._Ns = Ns

    def resample(self, samples, weights, resample_alg='roulette'):
        emp_mean = np.dot(weights, samples)
        emp_cov = 0
        for i in range(self._Ns):
            err = samples[i] - emp_mean
            emp_cov += weights[i] * np.outer(err, err)
        emp_cov = (emp_cov + emp_cov.T) / 2
        D = cholcov(emp_cov, lower=True)

        sample, _ = disc_random(weights, self._Ns, samples, alg=resample_alg)
        sample = np.array(sample, dtype=float)
        weight = np.full(weights.shape, 1 / self._Ns, dtype=float)

        # sample from beta distribution
        beta = np.random.beta(self._dim / 2, 2, self._Ns)
        # sample from a uniform distribution over unit sphere
        r = np.random.rand(self._Ns)
        r = r**(1 / self._dim)      # cdf: r^(1/n)
        theta = np.random.randn(self._dim, self._Ns)
        theta = theta / lg.norm(theta, axis=0)       # normalize random vector
        T = r * theta
        # sample from epanechnikov kernal
        eps = np.sqrt(beta) * T
        sample[:] = sample + self.opt_bandwidth * np.dot(D, eps).T

        return sample, weight

    @staticmethod
    def unit_hypershpere_volumn(dim):
        if dim % 2 == 0:
            vol = np.pi
            n = 2
            while n < dim:
                n += 2
                vol = 2 * np.pi * vol / n
        else:
            vol = 2
            n = 1
            while n < dim:
                n += 2
                vol = 2 * np.pi * vol / n
        return vol


class GaussianKernal():
    def __init__(self, dim, Ns):
        n = dim + 4
        self.opt_bandwidth = (4 / (dim + 2) / Ns)**(1 / n)
        self._dim = dim
        self._Ns = Ns

    def resample(self, samples, weights, resample_alg='roulette'):
        emp_mean = np.dot(weights, samples)
        emp_cov = 0
        for i in range(self._Ns):
            err = samples[i] - emp_mean
            emp_cov += weights[i] * np.outer(err, err)
        emp_cov = (emp_cov + emp_cov.T) / 2
        D = cholcov(emp_cov, lower=True)

        sample, _ = disc_random(weights, self._Ns, samples, alg=resample_alg)
        sample = np.array(sample, dtype=float)
        weight = np.full(weights.shape, 1 / self._Ns, dtype=float)

        eps = np.random.randn(self._dim, self._Ns)
        sample[:] = sample + self.opt_bandwidth * np.dot(D, eps).T

        return sample, weight
