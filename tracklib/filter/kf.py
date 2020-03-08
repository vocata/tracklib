# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
from tracklib import utils

__all__ = ['KFilter', 'SeqKFilter']


class KFilter():
    '''
    Normal linear kalman filter

    system model:
    x_k = F_k-1*x_k-1 + G_k-1*u_k-1 + L_k-1*w_k-1
    z_k = H_k*x_k + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, x_dim, z_dim, w_dim, v_dim, u_dim=0, at=1):
        '''
        x_dim: state dimension
        z_dim: measurement dimension
        w_dim: process noise dimension
        v_dim: measurement noises dimension
        u_dim: input dimension
        at: attenuation factor
        '''

        self._x_dim = x_dim
        self._z_dim = z_dim
        self._w_dim = w_dim
        self._v_dim = v_dim
        self._u_dim = u_dim
        self._at = at;

        self._x_pred = np.empty((x_dim, 1))
        self._P_pred = np.empty((x_dim, x_dim))
        self._x_up = np.empty((x_dim, 1))
        self._P_up = np.empty((x_dim, x_dim))
        self._innov = np.empty((z_dim, 1))
        self._inP = np.empty((z_dim, z_dim))
        self._K = np.empty((x_dim, z_dim))

        self._x_init = np.empty((x_dim, 1))
        self._P_init = np.empty((x_dim, x_dim))

        # initiate relevant matrix
        self._F = np.empty((x_dim, x_dim))
        self._G = np.empty((x_dim, u_dim))
        self._L = np.empty((x_dim, w_dim))
        self._H = np.empty((z_dim, x_dim))
        self._M = np.empty((z_dim, v_dim))
        self._Q = np.empty((w_dim, w_dim))
        self._R = np.empty((v_dim, v_dim))

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

    def init(self, x_init, P_init, **kw):
        self._x_init[:] = x_init
        self._P_init[:] = P_init
        self._x_pred[:] = x_init
        self._P_pred[:] = P_init
        self._x_up[:] = x_init
        self._P_up[:] = P_init
        if len(kw) > 0:
            if 'F' in kw: self._F[:] = kw['F']
            if 'G' in kw: self._G[:] = kw['G']
            if 'L' in kw: self._L[:] = kw['L']
            if 'Q' in kw: self._Q[:] = kw['Q']
            if 'H' in kw: self._H[:] = kw['H']
            if 'M' in kw: self._M[:] = kw['M']
            if 'R' in kw: self._R[:] = kw['R']
            if 'at' in kw: self._at = kw['at']
        self._len = 0
        self._stage = 0

    def predict(self, u=None, **kw):
        assert (self._stage == 0)

        if len(kw) > 0:
            if 'F' in kw: self._F[:] = kw['F']
            if 'G' in kw: self._G[:] = kw['G']
            if 'L' in kw: self._L[:] = kw['L']
            if 'Q' in kw: self._Q[:] = kw['Q']

        Q_tilde = self._L @ self._Q @ self._L.T
        ctl = 0 if u is None else self._G @ u
        self._x_pred[:] = self._F @ self._x_up + ctl
        self._P_pred[:] = self._at**2 * self._F @ self._P_up @ self._F.T + Q_tilde
        self._P_pred[:] = (self._P_pred + self._P_pred.T) / 2
        self._stage = 1  # predict finished
        return self._x_pred, self._P_pred

    def update(self, z, **kw):
        assert (self._stage == 1)

        if len(kw) > 0:
            if 'H' in kw: self._H[:] = kw['H']
            if 'M' in kw: self._M[:] = kw['M']
            if 'R' in kw: self._R[:] = kw['R']

        R_tilde = self._M @ self._R @ self._M.T
        z_pred = self._H @ self._x_pred
        self._innov[:] = z - z_pred
        self._inP[:] = self._H @ self._P_pred @ self._H.T + R_tilde
        self._inP[:] = (self._inP + self._inP.T) / 2
        self._K[:] = self._P_pred @ self._H.T @ linalg.inv(self._inP)
        self._x_up[:] = self._x_pred + self._K @ self._innov
        # The Joseph-form covariance update is used for improved numerical
        temp = np.eye(self._x_dim) - self._K @ self._H
        self._P_up[:] = temp @ self._P_pred @ temp.T + self._K @ R_tilde @ self._K.T
        self._P_up[:] = (self._P_up + self._P_up.T) / 2

        self._len += 1
        self._stage = 0  # update finished
        return self._x_up, self._P_up, self._K, self._innov, self._inP

    def step(self, z, u=None, **kw):
        assert (self._stage == 0)

        pred_ret = self.predict(u, **kw)
        update_ret = self.update(z, **kw)
        return pred_ret + update_ret

    def init_info(self):
        return self._x_init, self._P_init

    def predict_info(self):
        return self._x_pred, self._P_pred

    def update_info(self):
        return self._x_up, self._P_up, self._K, self._innov, self._inP


class SeqKFilter():
    '''
    Sequential kalman filter

    system model:
    x_k = F_k-1*x_k-1 + G_k-1*u_k-1 + L_k-1*w_k-1
    z_k = H_k*x_k + M_k*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, x_dim, z_dim, w_dim, v_dim, u_dim=0):
        '''
        x_dim: state dimension
        z_dim: measurement dimension
        w_dim: process noise dimension
        v_dim: measurement noises dimension
        u_dim: input dimension

        default is scalar kalman filter
        '''

        self._x_dim = x_dim
        self._z_dim = z_dim
        self._u_dim = u_dim

        self._x_pred = np.empty((x_dim, 1))
        self._P_pred = np.empty((x_dim, x_dim))
        self._x_up = np.empty((x_dim, 1))
        self._P_up = np.empty((x_dim, x_dim))

        self._x_init = np.empty((x_dim, 1))
        self._P_init = np.empty((x_dim, x_dim))

        # initiate relevant matrix
        self._F = np.empty((x_dim, x_dim))
        self._G = np.empty((x_dim, u_dim))
        self._L = np.empty((x_dim, w_dim))
        self._H = np.empty((z_dim, x_dim))
        self._M = np.empty((z_dim, v_dim))
        self._Q = np.empty((w_dim, w_dim))
        self._R = np.empty((v_dim, v_dim))
        self._S = np.empty((z_dim, z_dim))
        self._D = np.empty((z_dim, z_dim))

        self._len = 0
        self._stage = 0

    def __len__(self):
        return self._len

    def __str__(self):
        msg = 'sequential linear kalman filter: \n'
        msg += 'predicted state:\n%s\n\n' % str(self._x_pred)
        msg += 'predicted error covariance matrix:\n%s\n\n' % str(self._P_pred)
        msg += 'updated state:\n%s\n\n' % str(self._x_up)
        msg += 'updated error covariance matrix:\n%s\n' % str(self._P_up)
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, x_init, P_init, **kw):
        self._x_init[:] = x_init
        self._P_init[:] = P_init
        self._x_pred[:] = x_init
        self._P_pred[:] = P_init
        self._x_up[:] = x_init
        self._P_up[:] = P_init
        if len(kw) > 0:
            if 'F' in kw: self._F[:] = kw['F']
            if 'G' in kw: self._G[:] = kw['G']
            if 'L' in kw: self._L[:] = kw['L']
            if 'Q' in kw: self._Q[:] = kw['Q']
            if 'H' in kw: self._H[:] = kw['H']
            if 'M' in kw: self._M[:] = kw['M']
            if 'R' in kw: self._R[:] = kw['R']
            if 'M' in kw and 'R' in kw:
                R_tilde = self._M @ self._R @ self._M.T
                v, self._S[:] = linalg.eigh(R_tilde)
                self._S[:] = self._S.T
                self._D[:] = np.diag(v)

        self._len = 0
        self._stage = 0

    def predict(self, u=None, **kw):
        assert (self._stage == 0)

        if len(kw) > 0:
            if 'F' in kw: self._F[:] = kw['F']
            if 'G' in kw: self._G[:] = kw['G']
            if 'L' in kw: self._L[:] = kw['L']
            if 'Q' in kw: self._Q[:] = kw['Q']

        Q_tilde = self._L @ self._Q @ self._L.T
        ctl = 0 if u is None else self._G @ u
        self._x_pred[:] = self._F @ self._x_up + ctl
        self._P_pred[:] = self._F @ self._P_up @ self._F.T + Q_tilde
        self._P_pred[:] = (self._P_pred + self._P_pred.T) / 2
        self._stage = 1  # predict finished
        return self._x_pred, self._P_pred

    def update(self, z, **kw):
        assert (self._stage == 1)

        if len(kw) > 0:
            if 'H' in kw: self._H[:] = kw['H']
            if 'M' in kw: self._M[:] = kw['M']
            if 'R' in kw: self._R[:] = kw['R']
            if 'M' in kw or 'R' in kw:
                R_tilde = self._M @ self._R @ self._M.T
                v, self._S[:] = linalg.eigh(R_tilde)
                self._S[:] = self._S.T
                self._D[:] = np.diag(v)

        x_up = self._x_pred
        P_up = self._P_pred
        H_tilde = self._S @ self._H
        z_tilde = self._S @ z
        for n in range(self._z_dim):
            H_n = utils.row(H_tilde[n, :])
            z_n = utils.col(z_tilde[n])
            r_n = self._D[n, n]

            z_pred = H_n @ x_up
            innov = z_n - z_pred
            inP = H_n @ P_up @ H_n.T + r_n
            K = (P_up @ H_n.T) / inP
            x_up = x_up + K @ innov
            # The Joseph-form covariance update is used for improved numerical
            temp = np.eye(self._x_dim) - K @ H_n
            P_up = temp @ P_up @ temp.T + r_n * K @ K.T
            P_up = (P_up + P_up.T) / 2
        self._x_up[:] = x_up
        self._P_up[:] = P_up

        self._len += 1
        self._stage = 0
        return self._x_up, self._P_up

    def step(self, z, u=None, **kw):
        assert (self._stage == 0)

        pred_ret = self.predict(u, **kw)
        update_ret = self.update(z, **kw)
        return pred_ret + update_ret
