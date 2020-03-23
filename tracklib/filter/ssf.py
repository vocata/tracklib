# -*- coding: utf-8 -*-
'''
The module includes various steady-state filters, including alpha filter, alpha-beta filter,
alpha-beta-gamma filter and general Kalman steady-state filter.
'''
from __future__ import division, absolute_import, print_function


__all__ = ['AlphaFilter', 'AlphaBetaFilter', 'AlphaBetaGammaFilter', 'SSFilter']

import numpy as np
import scipy.linalg as lg
from .kfbase import KFBase
from tracklib.model import newton_sys


class AlphaFilter(KFBase):
    '''
    Alpha filter(one-state Newtonian system)

    system model:
    x_k = F*x_k-1 + L*w_k-1
    z_k = H*x_k + v_k
    E(w_k*w_j') = Q*δ_kj
    E(v_k*v_j') = R*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other.
    w_k and v_k are WGN vector.
    Q and R must be diagonal matrix, this means
    that the state and measurement on each axis
    are independent of each other.r
    '''
    def __init__(self, T, axis, alpha):
        '''
        T: sample interval
        axis: the number of target motion direction,
              such as x(1), xy(2) or xyz(3)
        '''
        super().__init__()
        self._alpha = alpha
        self._gain = np.diag(self._alpha)

        self._T = T
        self._axis = axis
        self._F, _, self._H, _ = newton_sys(T, 1, axis)

    def __str__(self):
        msg = 'Alpha filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state):
        self._prior_state = state
        self._post_state = state
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        self._prior_state = self._F @ self._post_state

        self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        z_prior = self._H @ self._prior_state
        self._post_state = self._prior_state + self._gain @ (z - z_prior)

        self._len += 1
        self._stage = 0

    def step(self, z):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        self.predict()
        self.update(z)

    @staticmethod
    def cal_params(sigma_w, sigma_v, T):
        '''
        obtain alpha and for which alpha filter
        becomes a steady-state Kalman filter
        '''
        if isinstance(sigma_w, np.ndarray):
            pass
        elif isinstance(sigma_w, (int, float)):
            sigma_w = np.array([sigma_w], dtype=float)
        elif isinstance(sigma_w, (list, tuple)):
            sigma_w = np.array(sigma_w, dtype=float)
        else:
            raise TypeError(
                'sigma_w must be a int, float, list, tuple or ndarray, not %s' %
                sigma_w.__class__.__name__)
        if isinstance(sigma_v, np.ndarray):
            pass
        elif isinstance(sigma_v, (int, float)):
            sigma_v = np.array([sigma_v], dtype=float)
        elif isinstance(sigma_v, (list, tuple)):
            sigma_v = np.array(sigma_v, dtype=float)
        else:
            raise TypeError(
                'sigma_v must be a int, float, list, tuple or ndarray, not %s' %
                sigma_v.__class__.__name__)

        lamb = sigma_w * T**2 / sigma_v
        alpha = (-lamb**2 + np.sqrt(lamb**4 + 16 * lamb**2)) / 8
        return alpha

class AlphaBetaFilter(KFBase):
    '''
    Alpha-beta filter(two-state Newtonian system)

    system model:
    x_k = F*x_k-1 + L*w_k-1
    z_k = H*x_k + v_k
    E(w_k*w_j') = Q*δ_kj
    E(v_k*v_j') = R*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other.
    w_k and v_k are WGN vector.
    Q and R must be diagonal matrix, this means
    that the state and measurement on each axis
    are independent of each other.r
    '''
    def __init__(self, T, axis, alpha, beta):
        '''
        T: sample interval
        axis: the number of target motion direction,
              such as x(1), xy(2) or xyz(3)
        '''
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        diag_a, diag_b = map(np.diag, (self._alpha, self._beta))
        self._gain = np.vstack((diag_a, diag_b / self._T))

        self._T = T
        self._axis = axis
        self._F, _, self._H, _= newton_sys(T, 2, axis)

    def __str__(self):
        msg = 'Alpha-beta filter:\n\n'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state):
        self._prior_state = state
        self._post_state = state
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        self._prior_state = self._F @ self._post_state

        self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        z_prior = self._H @ self._prior_state
        self._post_state = self._prior_state + self._gain @ (z - z_prior)
        
        self._len += 1
        self._stage = 0

    def step(self, z):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        self.predict()
        self.update(z)

    @staticmethod
    def cal_params(sigma_w, sigma_v, T):
        '''
        obtain alpha, beta and for which alpha-beta
        filter becomes a steady-state Kalman filter
        '''
        if isinstance(sigma_w, np.ndarray):
            pass
        elif isinstance(sigma_w, (int, float)):
            sigma_w = np.array([sigma_w], dtype=float)
        elif isinstance(sigma_w, (list, tuple)):
            sigma_w = np.array(sigma_w, dtype=float)
        else:
            raise TypeError(
                'sigma_w must be a int, float, list, tuple or ndarray, not %s' %
                sigma_w.__class__.__name__)
        if isinstance(sigma_v, np.ndarray):
            pass
        elif isinstance(sigma_v, (int, float)):
            sigma_v = np.array([sigma_v], dtype=float)
        elif isinstance(sigma_v, (list, tuple)):
            sigma_v = np.array(sigma_v, dtype=float)
        else:
            raise TypeError(
                'sigma_v must be a int, float, list, tuple or ndarray, not %s' %
                sigma_v.__class__.__name__)

        lamb = sigma_w * T**2 / sigma_v
        r = (4 + lamb - np.sqrt(8 * lamb + lamb**2)) / 4
        alpha = 1 - r**2
        beta = 2 * (2 - alpha) - 4 * np.sqrt(1 - alpha)
        return alpha, beta


class AlphaBetaGammaFilter():
    '''
    Alpha-beta-gamma filter(three-state Newtonian system)

    system model:
    x_k = F*x_k-1 + L*w_k-1
    z_k = H*x_k + v_k
    E(w_k*w_j') = Q*δ_kj
    E(v_k*v_j') = R*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other.
    w_k and v_k are WGN vector.
    Q and R must be diagonal matrix, this means
    that the state and measurement on each axis
    are independent of each other.r
    '''
    def __init__(self, T, axis, alpha, beta, gamma):
        '''
        T: sample interval
        axis: the number of target motion direction,
              such as x(1), xy(2) or xyz(3)
        '''
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        diag_a, diag_b, diag_g = map(np.diag, (self._alpha, self._beta, self._gamma))
        self._gain = np.vstack((diag_a, diag_b / self._T, diag_g / (2 * self._T**2)))

        self._T = T
        self._axis = axis
        self._F, _, self._H, _ = newton_sys(T, 3, axis)

    def __str__(self):
        msg = 'Alpha-beta-gamma filter:\n\n'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state):
        self._prior_state = state
        self._post_state = state
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        self._prior_state = self._F @ self._post_state

        self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        z_prior = self._H @ self._prior_state
        self._post_state = self._prior_state + self._gain @ (z - z_prior)

        self._len += 1
        self._stage = 0

    def step(self, z):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        self.predict()
        self.update(z)

    @staticmethod
    def cal_params(sigma_w, sigma_v, T):
        '''
        obtain alpha, beta and gamma for which
        alpha-beta-gamma becomes a steady-state
        Kalman filter
        '''
        if isinstance(sigma_w, np.ndarray):
            pass
        elif isinstance(sigma_w, (int, float)):
            sigma_w = np.array([sigma_w], dtype=float)
        elif isinstance(sigma_w, (list, tuple)):
            sigma_w = np.array(sigma_w, dtype=float)
        else:
            raise TypeError(
                'sigma_w must be a int, float, list, tuple or ndarray, not %s' %
                sigma_w.__class__.__name__)
        if isinstance(sigma_v, np.ndarray):
            pass
        elif isinstance(sigma_v, (int, float)):
            sigma_v = np.array([sigma_v], dtype=float)
        elif isinstance(sigma_v, (list, tuple)):
            sigma_v = np.array(sigma_v, dtype=float)
        else:
            raise TypeError(
                'sigma_v must be a int, float, list, tuple or ndarray, not %s' %
                sigma_v.__class__.__name__)
                
        lamb = sigma_w * T**2 / sigma_v
        b = lamb / 2 - 3
        c = lamb / 2 + 3
        d = -1
        p = c - b**2 / 3
        q = 2 * b**3 / 27 - b * c / 3 + d
        v = np.sqrt(q**2 + 4 * p**3 / 27)
        z = -np.cbrt(q + v / 2)
        s = z - p / (3 * z) - b / 3
        alpha = 1 - s**2
        beta = 2 * (1 - s)**2
        gamma = beta**2 / (2 * alpha)
        return alpha, beta, gamma


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
    def __init__(self, F, L, H, M, Q, R, G=None):
        super().__init__()

        self._F = F
        self._L = L
        self._H = H
        self._M = M
        self._Q = Q
        self._R = R
        self._G = G

    def __str__(self):
        msg = 'Steady-state linear Kalman filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov, it=5):
        self._state = state
        self._cov = cov
        self._prior_state = state
        self._post_state = state
        self._gain, self._prior_cov, self._post_cov = \
            SSFilter.issv(cov, self._F, self._L, self._H, self._M, self._Q, self._R, it=it)
        R_tilde = self._M @ self._R @ self._M.T
        self._innov_cov = self._H @ self._prior_cov @ self._H.T + R_tilde
        self._innov_cov = (self._innov_cov + self._innov_cov.T) / 2
        self._len = 0
        self._stage = 0
        self._init = True

    def predict(self, u=None):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        ctl = 0 if u is None else self._G @ u
        self._prior_state = self._F @ self._post_state + ctl

        self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        z_prior = self._H @ self._prior_state
        self._innov = z - z_prior
        self._post_state = self._prior_state + self._gain @ self._innov

        self._stage = 0
        self._len += 1

    def step(self, z, u=None):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        self.predict(u)
        self.update(z)

    @staticmethod
    def issv(P, F, L, H, M, Q, R, it):
        '''
        obtain Kalman filter steady-state value using iterative method
        note: "it" value can not be too large or it will diverge
        '''
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
        K = prior_cov @ H.T @ lg.inv(H @ prior_cov @ H.T + R_hat)
        post_cov = (I - K @ H) @ prior_cov @ (I - K @ H).T + K @ R_hat @ K.T

        return K, prior_cov, post_cov
