# -*- coding: utf-8 -*-
'''
Steady-state Kalman filter
This file contains alpha, alpha-beta, alpha-beta-gamma filter and generic steady-state Kalman filter,
of which only generic steady-state Kalman filter can be used for multi-model filtering.

REFERENCE:
[1]. D. Simon, "Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches," John Wiley and Sons, Inc., 2006.
[2]. P. R. Kalata, "The Tracking Index: A Generalized Parameter for α-β and α-β-γ Target Trackers," in IEEE Transactions on Aerospace and Electronic Systems, vol. AES-20, no. 2, pp. 174-182, March 1984.
[3]. Bar-Shalom Y., Li, X. R. and Kirubarajan, T, "Estimation with Applications to Tracking and Navigation," New York: John Wiley & Sons, 2001.
'''
from __future__ import division, absolute_import, print_function


__all__ = [
    'get_alpha', 'AlphaFilter', 'get_alpha_beta', 'AlphaBetaFilter',
    'get_alpha_beta_gamma', 'AlphaBetaGammaFilter', 'numerical_ss',
    'analytic_ss', 'SSFilter'
]

import numpy as np
import scipy.linalg as lg
from functools import reduce
from .base import KFBase
from tracklib.model import F_poly, H_pos_only


def get_alpha(sigma_w, sigma_v, T):
    '''
    Obtain alpha and for which alpha filter becomes a steady-state Kalman filter
    '''
    sigma_w = np.array(sigma_w, dtype=float)
    sigma_v = np.array(sigma_v, dtype=float)

    lamb = sigma_w * T**2 / sigma_v
    alpha = (-lamb**2 + np.sqrt(lamb**4 + 16 * lamb**2)) / 8
    return alpha


class AlphaFilter(KFBase):
    '''
    Alpha filter(one-state Newtonian system)

    system model:
    x_k = F*x_k-1 + w_k-1
    z_k = H*x_k + v_k
    E(w_k*w_j') = Q*δ_kj
    E(v_k*v_j') = R*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other.
    w_k and v_k are WGN vector.
    Q and R must be diagonal matrix, this means
    that the state and measurement on each axis
    are independent of each other.r
    '''
    def __init__(self, alpha, xdim, zdim, T):
        '''
        alpha : 1-D array-like, of length zdim
        xdim : int
        zdim : int
        T : float
            The time-duration of the propagation interval.
        '''
        super().__init__()
        self._alpha = alpha
        self._xdim = xdim
        self._wdim = zdim
        self._zdim = zdim
        self._vdim = zdim
        self._gain = np.diag(self._alpha)

        order = xdim // zdim
        axis = zdim
        self._F = F_poly(order, axis, T)
        self._H = H_pos_only(order, axis)

    def __str__(self):
        msg = 'Alpha filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state):
        self._post_state = state.copy()
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self._prior_state = self._F @ self._post_state

        self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        z_prior = self._H @ self._prior_state
        self._innov = z - z_prior
        self._post_state = self._prior_state + self._gain @ self._innov

        self._len += 1
        self._stage = 0

    def step(self, z):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict()
        self.update(z)


def get_alpha_beta(sigma_w, sigma_v, T):
    '''
    Obtain alpha, beta and for which alpha-beta filter becomes a steady-state Kalman filter
    '''
    sigma_w = np.array(sigma_w, dtype=float)
    sigma_v = np.array(sigma_v, dtype=float)

    lamb = sigma_w * T**2 / sigma_v
    r = (4 + lamb - np.sqrt(8 * lamb + lamb**2)) / 4
    alpha = 1 - r**2
    beta = 2 * (2 - alpha) - 4 * np.sqrt(1 - alpha)
    return alpha, beta


class AlphaBetaFilter(KFBase):
    '''
    Alpha-beta filter(two-state Newtonian system)

    system model:
    x_k = F*x_k-1 + w_k-1
    z_k = H*x_k + v_k
    E(w_k*w_j') = Q*δ_kj
    E(v_k*v_j') = R*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other.
    w_k and v_k are WGN vector.
    Q and R must be diagonal matrix, this means
    that the state and measurement on each axis
    are independent of each other.r
    '''
    def __init__(self, alpha, beta, xdim, zdim, T):
        '''
        alpha : 1-D array-like, of length zdim
        beta : 1-D array-like, of length zdim
        xdim : int
        zdim : int
        T : float
            The time-duration of the propagation interval.
        '''
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._xdim = xdim
        self._wdim = zdim
        self._zdim = zdim
        self._vdim = zdim

        trans = lambda x: np.array(x, dtype=float).reshape(-1, 1)
        block = [trans([alpha[i], beta[i] / T]) for i in range(zdim)]
        self._gain = lg.block_diag(*block)

        order = xdim // zdim
        axis = zdim
        self._F = F_poly(order, axis, T)
        self._H = H_pos_only(order, axis)

    def __str__(self):
        msg = 'Alpha-beta filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state):
        self._post_state = state.copy()
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self._prior_state = self._F @ self._post_state

        self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        z_prior = self._H @ self._prior_state
        self._innov = z - z_prior
        self._post_state = self._prior_state + self._gain @ self._innov

        self._len += 1
        self._stage = 0

    def step(self, z):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict()
        self.update(z)


def get_alpha_beta_gamma(sigma_w, sigma_v, T):
    '''
    obtain alpha, beta and gamma for which alpha-beta-gamma becomes a steady-state Kalman filter
    '''
    sigma_w = np.array(sigma_w, dtype=float)
    sigma_v = np.array(sigma_v, dtype=float)

    lamb = sigma_w * T**2 / sigma_v
    b = lamb / 2 - 3
    c = lamb / 2 + 3
    d = -1
    p = c - b**2 / 3
    q = 2 * b**3 / 27 - b * c / 3 + d
    v = np.sqrt(q**2 + 4 * p**3 / 27)
    z = -np.cbrt((q + v) / 2)
    s = z - p / (3 * z) - b / 3
    alpha = 1 - s**2
    beta = 2 * (1 - s)**2
    gamma = beta**2 / alpha

    return alpha, beta, gamma


class AlphaBetaGammaFilter(KFBase):
    '''
    Alpha-beta-gamma filter(three-state Newtonian system)

    system model:
    x_k = F*x_k-1 + w_k-1
    z_k = H*x_k + v_k
    E(w_k*w_j') = Q*δ_kj
    E(v_k*v_j') = R*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other.
    w_k and v_k are WGN vector.
    Q and R must be diagonal matrix, this means
    that the state and measurement on each axis
    are independent of each other.r
    '''
    def __init__(self, alpha, beta, gamma, xdim, zdim, T):
        '''
        alpha : 1-D array-like, of length zdim
        beta : 1-D array-like, of length zdim
        gamma : 1-D array-like, of length zdim
        xdim : int
        zdim : int
        T : float
            The time-duration of the propagation interval.
        '''
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._xdim = xdim
        self._wdim = zdim
        self._zdim = zdim
        self._vdim = zdim

        trans = lambda x: np.array(x, dtype=float).reshape(-1, 1)
        block = [trans([alpha[i], beta[i] / T, gamma[i] / (2 * T**2)]) for i in range(zdim)]
        self._gain = lg.block_diag(*block)

        order = xdim // zdim
        axis = zdim
        self._F = F_poly(order, axis, T)
        self._H = H_pos_only(order, axis)

    def __str__(self):
        msg = 'Alpha-beta-gamma filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state):
        self._post_state = state.copy()
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self._prior_state = self._F @ self._post_state

        self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        z_prior = self._H @ self._prior_state
        self._innov = z - z_prior
        self._post_state = self._prior_state + self._gain @ self._innov

        self._len += 1
        self._stage = 0

    def step(self, z):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict()
        self.update(z)



def numerical_ss(P, F, L, H, M, Q, R, it=5):
    '''
    obtain numerical Kalman filter steady-state quantities using iterative method
    note: 'it' can not be too large or it will diverge
    '''
    P = F @ P @ F.T + L @ Q @ L.T
    F_inv = lg.inv(F)
    Q_hat = L @ Q @ L.T
    R_hat = M @ R @ M.T
    R_inv = lg.inv(R_hat)
    lt = F + Q_hat @ F_inv.T @ H.T @ R_inv @ H
    rt = Q_hat @ F_inv.T
    lb = F_inv.T @ H.T @ R_inv @ H
    rb = F_inv.T
    top = np.hstack((lt, rt))
    bottom = np.hstack((lb, rb))
    psi = np.vstack((top, bottom))
    for _ in range(it):
        np.matmul(psi, psi, out=psi)
    I = np.eye(*P.shape)
    tmp = psi @ np.vstack((P, I))
    A_inf = tmp[:P.shape[0], :]
    B_inf = tmp[P.shape[0]:, :]

    prior_cov = A_inf @ lg.inv(B_inf)
    prior_cov = (prior_cov + prior_cov.T) / 2
    innov_cov = H @ prior_cov @ H.T + R_hat
    innov_cov = (innov_cov + innov_cov.T) / 2
    gain = prior_cov @ H.T @ lg.inv(innov_cov)
    post_cov = prior_cov - gain @ innov_cov @ gain.T
    post_cov = (post_cov + post_cov.T) / 2

    return prior_cov, post_cov, innov_cov, gain

def analytic_ss(F, L, H, M, Q, R):
    '''
    obtain analytic Kalman filter steady-state quantities by solving discrete-time algebraic Riccati equation 
    '''
    # lg.solve_discrete_are()
    Q_hat = L @ Q @ L.T
    R_hat = M @ R @ M.T
    prior_cov = lg.solve_discrete_are(F.T, H.T, Q_hat, R_hat)
    prior_cov = (prior_cov + prior_cov.T) / 2
    innov_cov = H @ prior_cov @ H.T + R_hat
    innov_cov = (innov_cov + innov_cov.T) / 2
    gain = prior_cov @ H.T @ lg.inv(innov_cov)
    post_cov = prior_cov - gain @ innov_cov @ gain.T
    post_cov = (post_cov + post_cov.T) / 2

    return prior_cov, post_cov, innov_cov, gain

class SSFilter(KFBase):
    '''
    Steady-state Kalman filter for multiple state systems

    system model:
    x_k = F*x_k-1 + G*u_k-1 + L*w_k-1
    z_k = H*x_k + M*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, F, L, H, M, Q, R, xdim, zdim, G=None, alg='riccati'):
        super().__init__()

        self._F = F.copy()
        self._L = L.copy()
        self._H = H.copy()
        self._M = M.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._xdim = xdim
        self._wdim = self._Q.shape[0]
        self._zdim = zdim
        self._vdim = self._R.shape[0]
        if G is None:
            self._G = G
        else:
            self._G = G.copy()
        if alg == 'riccati' or alg == 'iterative':
            self._alg = alg
        else:
            raise ValueError('unknown algorithem: %s' % alg)

    def __str__(self):
        msg = 'Steady-state linear Kalman filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        self._cov = cov.copy()
        self._post_state = state.copy()
        if self._alg == 'riccati':
            self._prior_cov, self._post_cov, self._innov_cov, self._gain = analytic_ss(
                self._F, self._L, self._H, self._M, self._Q, self._R)
        else:
            self._prior_cov, self._post_cov, self._innov_cov, self._gain = numerical_ss(
                cov, self._F, self._L, self._H, self._M, self._Q, self._R)

        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        ctl = 0 if u is None else self._G @ u
        self._prior_state = self._F @ self._post_state + ctl

        self._stage = 1

    def update(self, z, **kw):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        z_prior = self._H @ self._prior_state
        self._innov = z - z_prior
        self._post_state = self._prior_state + self._gain @ self._innov

        self._stage = 0
        self._len += 1

    def step(self, z, u=None, **kw):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict(u, **kw)
        self.update(z, **kw)
