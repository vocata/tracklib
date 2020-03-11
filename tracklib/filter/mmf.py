# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy.linalg as linalg

class MMFilter():
    '''
    Multiple Model linear kalman filter

    system model:
    x_k = F*x_k-1 + G*u_k-1 + L*w_k-1
    z_k = H*x_k + M*v_k
    E(w_k*w_j') = Q_k*δ_kj
    E(v_k*v_j') = R_k*δ_kj

    w_k, v_k, x_0 are uncorrelated to each other
    '''
    def __init__(self, F, L, H, M, Q, R, G=None, prob=None):
        model_n = len(F)
        self._model_n = model_n
        if prob is None:
            self._prob = [1. / model_n] * model_n
        else:
            self._prob = prob

        self._x_pred = None
        self._P_pred = None
        self._x_up = None
        self._P_up = None
        self._innov = None
        self._inP = None
        self._K = None

        self._x_init = None
        self._P_init = None

        # initiate relevant matrix
        self._F = F
        self._G = G
        self._L = L
        self._H = H
        self._M = M
        self._Q = Q
        self._R = R

        self._len = 0
        self._stage = 0

    def __len__(self):
        return self._len

    def __str__(self):
        return ''

    def __repr(self):
        return self.__str__()

    def init(self, x_init, P_init):
        self._x_init = x_init
        self._P_init = P_init
        self._x_pred = x_init
        self._P_pred = P_init
        self._x_up = x_init
        self._P_up = P_init
        self._innov = [None] * self._model_n
        self._inP = [None] * self._model_n
        self._K = [None] * self._model_n

        self._len = 0
        self._stage = 0

    def predict(self, u=None):
        assert (self._stage == 0)

        for i in range(self._model_n):
            Q_tilde = self._L[i] @ self._Q[i] @ self._L.T[i]
            ctl = 0 if u is None else self._G[i] @ u
            self._x_pred[i] = self._F[i] @ self._x_up[i] + ctl
            self._P_pred[i] = self._F[i] @ self._P_up[i] @ self._F.T[i] + Q_tilde
            self._P_pred[i] = (self._P_pred[i] + self._P_pred[i].T) / 2
            self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        
        pdf = []
        for i in range(self._model_n):
            R_tilde = self._M[i] @ self._R[i] @ self._M[i].T
            z_pred=  self._M[i] @ self._x_pred[i]
            self._innov[i] = z - z_pred
            self._inP[i] = self._H[i] @ self._P_pred[i] @ self._H[i].T + R_tilde
            self._inP[i] = (self._inP[i] + self._inP[i].T) / 2
            self._K[i] = self._P_pred[i] @ self._H[i].T @ linalg.inv(self._inP[i])
            self._x_up[i] = self._x_pred[i] + self._K[i] @ self._innov[i]
            temp = np.eye(*self._F[i].shape) - self._K[i] @ self._H[i]
            self._P_up[i] = temp @ self._P_pred[i] @ temp.T + self._K[i] @ R_tilde @ self._K[i].T
            self._P_up[i] = (self._P_up[i] + self._P_up[i].T) / 2
            pdf.append((np.exp(-self._innov[i].T @ linalg.inv(self._inP[i]) @ self._innov[i] / 2) / \
                np.sqrt(linalg.det(2 * math.pi * self._inP[i]))).item())

        # Total Probability
        total = 0
        for i in range(self._model_n):
            total += pdf[i] * self._prob[i]
        # update all model posterior probability
        for i in range(self._model_n):
            self._prob[i] = pdf[i] * self._prob[i] / total
        # compute the weighted state estimate
        x_weight = np.zeros((self._x_dim, 1))
        for i in range(self._model_n):
            x_weight += self._prob[i] * self._x_up[i]

        self._len += 1
        self._stage = 0
        return x_weight

    def step(self, z, u=None):
        assert (self._stage == 0)
        self.predict(u)
        return self.update(z)

        