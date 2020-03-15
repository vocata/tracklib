# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
from tracklib.utils import col
from tracklib.math import num_diff, num_diff_hessian

__all__ = ['EKFilter_1st', 'EKFilter_2ed']


class EKFilter_1st():
    '''
    First-order extended kalman filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1, w_k-1)
    z_k = h_k(x_k, v_k)
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, at=1):
        self._at = at

        self._x_pred = None
        self._P_pred = None
        self._x_up = None
        self._P_up = None
        self._innov = None
        self._inP = None
        self._K = None

        self._x_init = None
        self._P_init = None

        self._len = 0
        self._stage = 0

    def __len__(self):
        return self._len

    def __str__(self):
        msg = 'Fisrt-order extended kalman filter:\n\n'
        msg += 'predicted state:\n%s\n\n' % str(self._x_pred)
        msg += 'predicted error covariance matrix:\n%s\n\n' % str(self._P_pred)
        msg += 'updated state:\n%s\n\n' % str(self._x_up)
        msg += 'updated error covariance matrix:\n%s\n\n' % str(self._P_up)
        msg += 'kalman gain:\n%s\n' % str(self._K)
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, x_init, P_init):
        self._x_init = x_init
        self._P_init = P_init
        self._x_pred = x_init
        self._P_pred = P_init
        self._x_up = x_init
        self._P_up = P_init
        self._len = 0
        self._stage = 0

    def predict(self, u, f, Q, F=None, L=None):
        assert (self._stage == 0)

        x_dim = self._x_init.shape[0]
        w_dim = Q.shape[0]
        if F is None:
            fx = lambda x: f(x, u, np.zeros((w_dim, 1)))
            F = num_diff(self._x_up, fx, x_dim)
        if L is None:
            fw = lambda w: f(self._x_up, u, w)
            L = num_diff(np.zeros((w_dim, 1)), fw, x_dim)

        Q_tilde = L @ Q @ L.T
        self._x_pred = f(self._x_up, u, 0)
        self._P_pred = self._at**2 * F @ self._P_up @ F.T + Q_tilde
        self._P_pred = (self._P_pred + self._P_pred.T) / 2

        self._stage = 1
        return self._x_pred, self._P_pred

    def update(self, z, h, R, H=None, M=None, it=0):
        assert (self._stage == 1)

        z_dim = z.shape[0]
        v_dim = R.shape[0]
        if not (H is None and M is None):
            it = 0
        if H is None:
            hx = lambda x: h(x, np.zeros((v_dim, 1)))
            H = num_diff(self._x_pred, hx, z_dim)
        if M is None:
            hv = lambda v: h(self._x_pred, v)
            M = num_diff(np.zeros((v_dim, 1)), hv, z_dim)

        R_tilde = M @ R @ M.T
        z_pred = h(self._x_pred, 0)
        self._innov = z - z_pred
        self._inP = H @ self._P_pred @ H.T + R_tilde
        self._inP = (self._inP + self._inP.T) / 2
        self._K = self._P_pred @ H.T @ lg.inv(self._inP)
        self._x_up = self._x_pred + self._K @ self._innov
        # The Joseph-form covariance update is used for improved numerical
        temp = np.eye(*self._P_pred.shape) - self._K @ H
        self._P_up = temp @ self._P_pred @ temp.T + self._K @ R_tilde @ self._K.T
        self._P_up = (self._P_up + self._P_up.T) / 2

        for _ in range(it):
            hx = lambda x: h(x, np.zeros((v_dim, 1)))
            H = num_diff(self._x_up, hx, z_dim)
            hv = lambda v: h(self._x_up, v)
            M = num_diff(np.zeros((v_dim, 1)), hv, z_dim)

            R_tilde = M @ R @ M.T
            z_pred = h(self._x_up, 0) + H @ (self._x_pred - self._x_up)
            self._innov = z - z_pred
            self._inP = H @ self._P_pred @ H.T + R_tilde
            self._inP = (self._inP + self._inP.T) / 2
            self._K = self._P_pred @ H.T @ lg.inv(self._inP)
            self._x_up = self._x_pred + self._K @ self._innov
            temp = np.eye(*self._P_pred.shape) - self._K @ H
            self._P_up = temp @ self._P_pred @ temp.T + self._K @ R_tilde @ self._K.T
            self._P_up = (self._P_up + self._P_up.T) / 2

        self._len += 1
        self._stage = 0
        return self._x_up, self._P_up, self._K, self._innov, self._inP

    def step(self, u, z, f, h, Q, R, F=None, L=None, H=None, M=None, it=0):
        assert (self._stage == 0)

        pred_ret = self.predict(u, f, Q, F=F, L=L)
        update_ret = self.update(z, h, R, H=H, M=M, it=it)
        return pred_ret + update_ret


class EKFilter_2ed():
    '''
    Second-order extended kalman filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1, w_k-1)
    z_k = h_k(x_k, v_k)
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, at=1):
        self._at = at

        self._x_pred = None
        self._P_pred = None
        self._x_up = None
        self._P_up = None
        self._innov = None
        self._inP = None
        self._K = None

        self._x_init = None
        self._P_init = None

        self._len = 0
        self._stage = 0

    def __len__(self):
        return self._len

    def __str__(self):
        msg = 'Second-order kalman filter:\n\n'
        msg += 'predicted state:\n%s\n\n' % str(self._x_pred)
        msg += 'predicted error covariance matrix:\n%s\n\n' % str(self._P_pred)
        msg += 'updated state:\n%s\n\n' % str(self._x_up)
        msg += 'updated error covariance matrix:\n%s\n\n' % str(self._P_up)
        msg += 'kalman gain:\n%s\n' % str(self._K)
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, x_init, P_init):
        self._x_init = x_init
        self._P_init = P_init
        self._x_pred = x_init
        self._P_pred = P_init
        self._x_up = x_init
        self._P_up = P_init
        self._len = 0
        self._stage = 0

    def predict(self, u, f, Q, F=None, L=None, f_Hess=None):
        assert (self._stage == 0)

        x_dim = self._x_init.shape[0]
        w_dim = Q.shape[0]
        if F is None:
            fx = lambda x: f(x, u, np.zeros((w_dim, 1)))
            F = num_diff(self._x_up, fx, x_dim)
        if L is None:
            fw = lambda w: f(self._x_up, u, w)
            L = num_diff(np.zeros((w_dim, 1)), fw, x_dim)
        if f_Hess is None:
            fx = lambda x: f(x, u, np.zeros((w_dim, 1)))
            f_Hess = num_diff_hessian(self._x_up, fx, x_dim)

        Q_tilde = L @ Q @ L.T
        self._x_pred = f(self._x_up, u, 0)
        quad = col([np.trace(f_Hess[:, :, i] @ self._P_up) for i in range(x_dim)])
        self._x_pred += quad / 2
        self._P_pred = self._at**2 * F @ self._P_up @ F.T + Q_tilde
        self._P_pred = (self._P_pred + self._P_pred.T) / 2

        self._stage = 1
        return self._x_pred, self._P_pred

    def update(self, z, h, R, H=None, M=None, h_Hess=None, it=0):
        assert (self._stage == 1)

        z_dim = z.shape[0]
        v_dim = R.shape[0]
        if not (H is None and M is None and H is None):
            it = 0
        if H is None:
            hx = lambda x: h(x, np.zeros((v_dim, 1)))
            H = num_diff(self._x_pred, hx, z_dim)
        if M is None:
            hv = lambda v: h(self._x_pred, v)
            M = num_diff(np.zeros((v_dim, 1)), hv, z_dim)
        if h_Hess is None:
            hx = lambda x: h(x, np.zeros((v_dim, 1)))
            h_Hess = num_diff_hessian(self._x_pred, hx, z_dim)

        R_tilde = M @ R @ M.T
        quad = col([np.trace(h_Hess[:, :, i] @ self._P_pred) for i in range(z_dim)])
        z_pred = h(self._x_pred, 0) + quad / 2
        self._innov = z - z_pred
        self._inP = H @ self._P_pred @ H.T + R_tilde
        self._inP = (self._inP + self._inP.T) / 2
        self._K = self._P_pred @ H.T @ lg.inv(self._inP)
        self._x_up = self._x_pred + self._K @ self._innov
        # The Joseph-form covariance update is used for improved numerical
        temp = np.eye(*self._P_pred.shape) - self._K @ H
        self._P_up = temp @ self._P_pred @ temp.T + self._K @ R_tilde @ self._K.T
        self._P_up = (self._P_up + self._P_up.T) / 2

        for _ in range(it):
            hx = lambda x: h(x, np.zeros((v_dim, 1)))
            H = num_diff(self._x_up, hx, z_dim)
            h_Hess = num_diff_hessian(self._x_up, hx, z_dim)
            hv = lambda v: h(self._x_up, v)
            M = num_diff(np.zeros((v_dim, 1)), hv, z_dim)

            R_tilde = M @ R @ M.T
            quad = col([np.trace(h_Hess[:, :, i] @ self._P_up) for i in range(z_dim)])
            z_pred = h(self._x_up, 0) + H @ (self._x_pred - self._x_up) + quad / 2
            self._innov = z - z_pred
            self._inP = H @ self._P_pred @ H.T + R_tilde
            self._inP = (self._inP + self._inP.T) / 2
            self._K = self._P_pred @ H.T @ lg.inv(self._inP)
            self._x_up = self._x_pred + self._K @ self._innov
            temp = np.eye(*self._P_pred.shape) - self._K @ H
            self._P_up = temp @ self._P_pred @ temp.T + self._K @ R_tilde @ self._K.T
            self._P_up = (self._P_up + self._P_up.T) / 2

        self._len += 1
        self._stage = 0
        return self._x_up, self._P_up, self._K, self._innov, self._inP

    def step(self, u, z, f, h, Q, R, F=None, L=None, H=None, M=None, f_Hess=None, h_Hess=None, it=0):
        assert (self._stage == 0)

        pred_ret = self.predict(u, f, Q, F=F, L=L, f_Hess=f_Hess)
        update_ret = self.update(z, h, R, H=H, M=M, h_Hess=h_Hess, it=it)
        return pred_ret + update_ret