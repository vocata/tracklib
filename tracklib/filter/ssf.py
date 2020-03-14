# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
from tracklib import utils
from .model import newton_sys
'''
steady-state kalman filter
'''

__all__ = ['AlphaFilter', 'AlphaBetaFilter', 'AlphaBetaGammaFilter', 'SSFilter']


class AlphaFilter():
    '''
    alpha filter(one-state Newtonian system)

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
        self._x_pred = None
        self._x_up = None

        self._x_init = None

        self._alpha = alpha
        diag_a = np.diag(self._alpha)
        self._K = diag_a

        self._T = T
        self._axis = axis
        self._F, _, self._H, _ = newton_sys(T, 1, axis)

        self._len = 0
        self._stage = 0

    def __len__(self):
        return self._len

    def __str__(self):
        msg = 'alpha filter: \n'
        msg += 'predicted state:\n%s\n\n' % str(self._x_pred)
        msg += 'updated state:\n%s\n\n' % str(self._x_up)
        msg += 'alpha: %s\n\n' % str(self._alpha)
        msg += 'kalman filter gain:\n%s\n' % str(self._K)
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, x_init):
        self._x_init = x_init
        self._x_pred = x_init
        self._x_up = x_init

        self._len = 0
        self._stage = 0

    def predict(self):
        assert (self._stage == 0)

        self._x_pred = self._F @ self._x_up
        self._stage = 1
        return self._x_pred

    def update(self, z):
        assert (self._stage == 1)

        self._x_up = self._x_pred + self._K @ (z - self._H @ self._x_pred)
        self._len += 1
        self._stage = 0
        return self._x_up

    def step(self, z):
        assert (self._stage == 0)
        return self.predict(), self.update(z)

    def steady_params(self):
        return self._K

    @staticmethod
    def cal_params(sigma_w, sigma_v, T):
        '''
        obtain alpha and for which alpha filter
        becomes a steady-state kalman filter
        '''
        sigma_w = utils.col(sigma_w)
        sigma_v = utils.col(sigma_v)
        lamb = sigma_w * T**2 / sigma_v
        alpha = (-lamb**2 + np.sqrt(lamb**4 + 16 * lamb**2)) / 8
        return alpha

class AlphaBetaFilter():
    '''
    alpha-beta filter(two-state Newtonian system)

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
        self._x_pred = None
        self._x_up = None

        self._x_init = None

        self._alpha = alpha
        self._beta = beta
        diag_a, diag_b = map(np.diag, (self._alpha, self._beta))
        self._K = np.vstack((diag_a, diag_b / self._T))

        self._T = T
        self._axis = axis
        self._F, _, self._H, _= newton_sys(T, 2, axis)

        self._len = 0
        self._stage = 0

    def __len__(self):
        return self._len

    def __str__(self):
        msg = 'alpha-beta filter: \n'
        msg += 'predicted state:\n%s\n\n' % str(self._x_pred)
        msg += 'updated state:\n%s\n\n' % str(self._x_up)
        msg += 'alpha: %s, beta: %s\n\n' % (str(self._alpha), str(self._beta))
        msg += 'kalman filter gain:\n%s\n' % str(self._K)
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, x_init):
        self._x_init = x_init
        self._x_pred = x_init
        self._x_up = x_init

        self._len = 0
        self._stage = 0

    def predict(self):
        assert (self._stage == 0)

        self._x_pred = self._F @ self._x_up
        self._stage = 1
        return self._x_pred

    def update(self, z):
        assert (self._stage == 1)

        self._x_up = self._x_pred + self._K @ (z - self._H @ self._x_pred)
        self._len += 1
        self._stage = 0
        return self._x_up

    def step(self, z):
        assert (self._stage == 0)
        return self.predict(), self.update(z)

    def steady_params(self):
        return self._K

    @staticmethod
    def cal_params(sigma_w, sigma_v, T):
        '''
        obtain alpha, beta and for which alpha-beta
        filter becomes a steady-state kalman filter
        '''
        sigma_w = utils.col(sigma_w)
        sigma_v = utils.col(sigma_v)
        lamb = sigma_w * T**2 / sigma_v
        r = (4 + lamb - np.sqrt(8 * lamb + lamb**2)) / 4
        alpha = 1 - r**2
        beta = 2 * (2 - alpha) - 4 * np.sqrt(1 - alpha)
        return alpha, beta


class AlphaBetaGammaFilter():
    '''
    alpha-beta-gamma filter(three-state Newtonian system)

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
        self._x_pred = None
        self._x_up = None

        self._x_init = None

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        diag_a, diag_b, diag_g = map(np.diag, (self._alpha, self._beta, self._gamma))
        self._K = np.vstack((diag_a, diag_b / self._T, diag_g / (2 * self._T**2)))

        self._T = T
        self._axis = axis
        self._F, _, self._H, _ = newton_sys(T, 3, axis)

        self._len = 0
        self._stage = 0

    def __len__(self):
        return self._len

    def __str__(self):
        msg = 'alpha-beta-gamma filter: \n'
        msg += 'predicted state:\n%s\n\n' % str(self._x_pred)
        msg += 'updated state:\n%s\n\n' % str(self._x_up)
        msg += 'alpha: %s, beta: %s, gamma: %s\n\n' % \
            (str(self._alpha), str(self._beta), str(self._gamma))
        msg += 'kalman filter gain:\n%s\n' % str(self._K)
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, x_init):
        self._x_init = x_init
        self._x_pred = x_init
        self._x_up = x_init

        self._len = 0
        self._stage = 0

    def predict(self):
        assert (self._stage == 0)

        self._x_pred = self._F @ self._x_up
        self._stage = 1
        return self._x_pred

    def update(self, z):
        assert (self._stage == 1)

        self._x_up[:] = self._x_pred + self._K @ (z - self._H @ self._x_pred)
        self._len += 1
        self._stage = 0
        return self._x_up

    def step(self, z):
        assert (self._stage == 0)
        return self.predict(), self.update(z)

    def steady_params(self):
        return self._K

    @staticmethod
    def cal_params(sigma_w, sigma_v, T):
        '''
        obtain alpha, beta and gamma for which
        alpha-beta-gamma becomes a steady-state
        kalman filter
        '''
        sigma_w = utils.col(sigma_w)
        sigma_v = utils.col(sigma_v)
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


class SSFilter():
    '''
    steady-state Kalman filter for multiple state systems

    system model:
    x_k = F*x_k-1 + G*u_k-1 + L*w_k-1
    z_k = H*x_k + M*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, F, L, H, M, Q, R, G=None):
        self._x_pred = None
        self._P_pred = None
        self._x_up = None
        self._P_up = None
        self._K = None

        self._x_init = None
        self._P_init = None

        # initiate relevant matrix
        self._F = F
        self._L = L
        self._H = H
        self._M = M
        self._Q = Q
        self._R = R
        self._G = G

        self._len = 0
        self._stage = 0

    def __len__(self):
        return self._len

    def __str__(self):
        msg = 'steady-state linear kalman filter: \n'
        msg += 'predicted state:\n%s\n\n' % str(self._x_pred)
        msg += 'steady-state predicted error covariance matrix:\n%s\n\n' % str(
            self._P_pred)
        msg += 'updated state:\n%s\n\n' % str(self._x_up)
        msg += 'steady-state updated error covariance matrix:\n%s\n\n' % str(
            self._P_up)
        msg += 'steady-state kalman filter gain:\n%s\n' % str(self._K)
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, x_init, P_init, it=5):
        self._x_init = x_init
        self._P_init = P_init
        self._x_pred = x_init
        self._x_up = x_init
        self._K, self._P_pred, self._P_up = \
            SSFilter.issv(P_init, self._F, self._L, self._H, self._M, self._Q, self._R, it=it)
        self._len = 0
        self._stage = 0

    def predict(self, u=None):
        assert (self._stage == 0)

        ctl = 0 if u is None else self._G @ u
        self._x_pred = self._F @ self._x_up + ctl
        self._stage = 1
        return self._x_pred

    def update(self, z):
        assert (self._stage == 1)

        z_pred = self._H @ self._x_pred
        self._x_up = self._x_pred + self._K @ (z - z_pred)
        self._stage = 0
        self._len += 1
        return self._x_up

    def step(self, z, u=None):
        assert (self._stage == 0)
        return self.predict(u), self.update(z)

    def steady_params(self):
        return self._K, self._P_pred, self._P_up

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
        P_pred = A_inf @ lg.inv(B_inf)
        K = P_pred @ H.T @ lg.inv(H @ P_pred @ H.T + R_hat)
        P_up = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R_hat @ K.T

        return K, P_pred, P_up
