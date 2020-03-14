# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
from tracklib.math import num_diff

__all__ = ['EKFilter_1st', 'EKFilter_2ed']


class EKFilter_1st():
    '''
    First order extended kalman filter

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
        msg = 'normal linear kalman filter: \n'
        msg += 'predicted state:\n%s\n\n' % str(self._x_pred)
        msg += 'predicted error covariance matrix:\n%s\n\n' % str(self._P_pred)
        msg += 'updated state:\n%s\n\n' % str(self._x_up)
        msg += 'updated error covariance matrix:\n%s\n\n' % str(self._P_up)
        msg += 'kalman filter gain:\n%s\n' % str(self._K)
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

        if F is None:
            fx = lambda x: f(x, u, np.zeros((Q.shape[0], 1)))
            FJacob = lambda x: num_diff(x, fx, self._x_init.shape[0])
            F = FJacob(self._x_up)
        if L is None:
            fw = lambda w: f(self._x_up, u, w)
            Ljacob = lambda w: num_diff(w, fw, self._x_init.shape[0])
            L = Ljacob(np.zeros((Q.shape[0], 1)))

        Q_tilde = L @ Q @ L.T
        self._x_pred = f(self._x_up, u, 0)
        self._P_pred = self._at**2 * F @ self._P_up @ F.T + Q_tilde
        self._P_pred = (self._P_pred + self._P_pred.T) / 2

        self._stage = 1
        return self._x_pred, self._P_pred

    def update(self, z, h, R, H=None, M=None, it=0):
        assert (self._stage == 1)

        if H is None:
            hx = lambda x: h(x, np.zeros((R.shape[0], 1)))
            Hjacob = lambda x: num_diff(x, hx, z.shape[0])
            H = Hjacob(self._x_pred)
        if M is None:
            hv = lambda v: h(self._x_pred, v)
            Mjacob = lambda v: num_diff(v, hv, z.shape[0])
            M = Mjacob(np.zeros((R.shape[0], 1)))

        R_tilde = M @ R @ M.T
        z_pred = h(self._x_pred, 0)
        self._innov = z - z_pred
        self._inP = H @ self._P_pred @ H.T + R_tilde
        self._inP = (self._inP + self._inP.T) / 2
        self._K = self._P_pred @ H.T @ lg.inv(self._inP)
        self._x_up = self._x_pred + self._K @ self._innov
        # The Joseph-form covariance update is used for improved numerical
        temp = np.eye(*self._P_init.shape) - self._K @ H
        self._P_up = temp @ self._P_pred @ temp.T + self._K @ R_tilde @ self._K.T
        self._P_up = (self._P_up + self._P_up.T) / 2

        for _ in range(it):
            hx = lambda x: h(x, np.zeros((R.shape[0], 1)))
            hv = lambda v: h(self._x_up, v)
            Hjacob = lambda x: num_diff(x, hx, z.shape[0])
            Mjacob = lambda v: num_diff(v, hv, z.shape[0])
            H = Hjacob(self._x_up)
            M = Mjacob(np.zeros((R.shape[0], 1)))

            R_tilde = M @ R @ M.T
            z_pred = h(self._x_up, 0) + H @ (self._x_pred - self._x_up)
            self._innov = z - z_pred
            self._inP = H @ self._P_pred @ H.T + R_tilde
            self._inP = (self._inP + self._inP.T) / 2
            self._K = self._P_pred @ H.T @ lg.inv(self._inP)
            self._x_up = self._x_pred + self._K @ self._innov
            temp = np.eye(*self._P_init.shape) - self._K @ H
            self._P_up = temp @ self._P_pred @ temp.T + self._K @ R_tilde @ self._K.T
            self._P_up = (self._P_up + self._P_up.T) / 2

        self._len += 1
        self._stage = 0
        return self._x_up, self._P_up, self._K, self._innov, self._inP

    def step(self, u, z, f, h, Q, R, F=None, L=None, H=None, M=None, it=0):
        assert (self._stage == 0)

        pred_ret = self.predict(u, f, Q, F=F, L=L)
        update_ret = self.update(z, h, R, H=H, M=M, it=0)
        return pred_ret + update_ret


class EKFilter_2ed():
    '''
    Second order extended kalman filter

    system model:
    x_k = f_k-1(x_k-1, u_k-1, w_k-1)
    z_k = h_k(x_k, v_k)
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    pass