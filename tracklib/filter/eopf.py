# -*- coding: utf-8 -*-
'''
SMC Extended object tracker

REFERENCE:
[1]. 
'''
from __future__ import division, absolute_import, print_function


__all__ = ['EOPFilter', 'EORBPFilter', 'TurnRateEORBPFilter']

import numpy as np
import scipy.linalg as lg
import scipy.stats as st
import scipy.special as sl
from .base import EOFilterBase
from tracklib.utils import ellip_volume, rotate_matrix_deg


class EOPFilter(EOFilterBase):
    '''
    Extended object particle filter
    '''
    def __init__(self, F, H, D, R, Ns, Neff, df, lamb=None):
        self._F = F.copy()
        self._H = H.copy()
        self._D = D.copy()
        self._R = R.copy()
        self._Ns = Ns
        self._Neff = Neff
        self._df = df
        self._lamb = lamb
        self._init = False

    def init(self, state, cov, df, extension):
        self._state = state.copy()
        self._cov = cov.copy()
        self._ext = extension.copy()

        # self._state_samples = st.multivariate_normal.rvs(state, cov, self._Ns)
        self._state_samples = np.full((self._Ns, state.shape[0]), state, dtype=float)
        self._ext_samples = st.wishart.rvs(df, extension / df, self._Ns)
        self._weights = np.full(self._Ns, 1 / self._Ns, dtype=float)
        self._init = True

    def predict(self):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        # update samples
        self._ext_samples[:] = [
            st.wishart.rvs(self._df, self._ext_samples[i] / self._df)
            for i in range(self._Ns)
        ]
        self._state_samples[:] = [
            st.multivariate_normal.rvs(np.dot(self._F, self._state_samples[i]),
                                       np.kron(self._ext_samples[i], self._D))
            for i in range(self._Ns)
        ]

        # compute prior extension, state and covariance
        self._state = 0
        self._ext = 0
        for i in range(self._Ns):
            self._state += self._weights[i] * self._state_samples[i]
            self._ext += self._weights[i] * self._ext_samples[i]
        self._ext = (self._ext + self._ext.T) / 2
        self._cov = 0
        for i in range(self._Ns):
            err = self._state_samples[i] - self._state
            self._cov += self._weights[i] * np.outer(err, err)
        self._cov = (self._cov + self._cov.T) / 2

        return self._state, self._cov, self._ext

    def correct(self, zs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        Nm = len(zs)    # measurements number
        if self._lamb is None:
            lamb = Nm / ellip_volume(self._ext)        # empirical target density
        else:
            lamb = self._lamb

        # update weights
        const_arr = np.zeros(self._Ns)
        dist_arr = np.zeros(self._Ns)
        for i in range(self._Ns):
            cov = self._ext_samples[i] / 4 + self._R
            V = ellip_volume(self._ext_samples[i])
            pmf = (lamb * V)**Nm * np.exp(-lamb * V) / sl.factorial(Nm)
            # pmf = st.poisson.pmf(Nm, lamb * V)    # too slow
            # pmf = 1
            const_arr[i] = pmf / lg.det(2 * np.pi * cov)**(Nm / 2)

            d = zs - np.dot(self._H, self._state_samples[i])
            dist = (lg.inv(cov) @ (d.T @ d)).trace() / 2
            # dist = 0
            # for j in range(Nm):
            #     d = zs[j] - np.dot(self._H, self._state_samples[i])
            #     dist += d @ cov_inv @ d / 2
            dist_arr[i] = dist
        # the underflow problem is avoided by adding offset to exp function
        exp_term = np.exp(-dist_arr + dist_arr.min())
        self._weights *= const_arr * exp_term
        self._weights /= self._weights.sum()

        # compute posterior extension, state and covariance
        self._state = 0
        self._ext = 0
        for i in range(self._Ns):
            self._state += self._weights[i] * self._state_samples[i]
            self._ext += self._weights[i] * self._ext_samples[i]
        self._ext = (self._ext + self._ext.T) / 2
        self._cov = 0
        for i in range(self._Ns):
            err = self._state_samples[i] - self._state
            self._cov += self._weights[i] * np.outer(err, err)
        self._cov = (self._cov + self._cov.T) / 2

        # resample
        Neff = 1 / (self._weights**2).sum()
        if Neff <= self._Neff:
            idx = np.random.choice(np.arange(self._Ns), p=self._weights, size=self._Ns)
            self._state_samples[:] = self._state_samples[idx]
            self._ext_samples[:] = self._ext_samples[idx]
            self._weights[:] = 1 / self._Ns

        return self._state, self._cov, self._ext

    def distance(self, zs, **kwargs):
        return super().distance(zs, **kwargs)

    def likelihood(self, zs, **kwargs):
        return super().likelihood(zs, **kwargs)


class EORBPFilter(EOFilterBase):
    '''
    Extended object Rao-Blackwellized particle filter
    '''
    def __init__(self, F, H, D, R, Ns, Neff, df, lamb=None):
        self._F = F.copy()
        self._H = H.copy()
        self._D = D.copy()
        self._R = R.copy()
        self._Ns = Ns
        self._Neff = Neff
        self._df = df
        self._lamb = lamb
        self._init = False

    def init(self, state, cov, df, extension):
        self._state = state.copy()
        self._cov = cov.copy()
        self._ext = extension.copy()

        # self._state_samples = st.multivariate_normal.rvs(state, cov, self._Ns)
        self._state_samples = np.full((self._Ns, state.shape[0]), state, dtype=float)
        self._cov_samples = np.full((self._Ns, cov.shape[0], cov.shape[1]), cov)
        self._ext_samples = st.wishart.rvs(df, extension / df, self._Ns)
        self._weights = np.full(self._Ns, 1 / self._Ns, dtype=float)
        self._init = True

    def predict(self):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        # update samples
        for i in range(self._Ns):
            self._state_samples[i] = np.dot(self._F, self._state_samples[i])
            self._cov_samples[i] = self._F @ self._cov_samples[i] @ self._F.T + np.kron(self._ext_samples[i], self._D)
            self._ext_samples[i] = st.wishart.rvs(self._df, self._ext_samples[i] / self._df)

        # compute prior extension, state and covariance
        self._state = 0
        self._cov = 0
        self._ext = 0
        for i in range(self._Ns):
            self._state += self._weights[i] * self._state_samples[i]
            self._cov += self._weights[i] * self._cov_samples[i]
            self._ext += self._weights[i] * self._ext_samples[i]
        self._cov = (self._cov + self._cov.T) / 2
        self._ext = (self._ext + self._ext.T) / 2

        return self._state, self._cov, self._ext

    def correct(self, zs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        Nm = len(zs)    # measurements number
        if self._lamb is None:
            lamb = Nm / ellip_volume(self._ext)        # empirical target density
        else:
            lamb = self._lamb

        H = np.kron(np.ones((Nm, 1)), self._H)      # expand dimension of H
        z_mean = np.mean(zs, axis=0)
        for i in range(self._Ns):
            # update weight
            V = ellip_volume(self._ext_samples[i])
            pmf = (lamb * V)**Nm * np.exp(-lamb * V) / sl.factorial(Nm)
            self._weights[i] *= pmf

            z_pred = np.dot(H, self._state_samples[i])
            R = self._ext_samples[i] / 4 + self._R
            # expanded dimension innovation covariance
            P = np.kron(np.eye(Nm), R) + H @ self._cov_samples[i] @ H.T
            if Nm <= 3:
                # calculate the inversion of P directly
                P_inv = lg.inv(P)
            else:
                # calculate the inversion of P using `matrix inversion lemma`
                A_inv = np.kron(np.eye(Nm), lg.inv(R))
                D = lg.inv(self._cov_samples[i])
                P_inv = A_inv - A_inv @ H @ lg.inv(D + H.T @ A_inv @ H) @ H.T @ A_inv
            # compute the likelihood of measurements, this is the key process
            z_vec = np.reshape(zs, -1)
            pdf = np.exp(-(z_vec - z_pred) @ P_inv @ (z_vec - z_pred) / 2) / np.sqrt(lg.det(2 * np.pi * P))
            self._weights[i] *= pdf
            # self._weights[i] *= st.multivariate_normal.pdf(z_vec, mean=z_pred, cov=P)     # too slow

            # update state and covariance using Kalman filter using mean measurement
            innov = z_mean - np.dot(self._H, self._state_samples[i])
            S = self._H @ self._cov_samples[i] @ self._H.T + R / Nm
            S = (S + S.T) / 2
            K = self._cov_samples[i] @ self._H.T @ lg.inv(S)

            self._state_samples[i] += np.dot(K, innov)
            self._cov_samples[i] -= K @ S @ K.T
            self._cov_samples[i] = (self._cov_samples[i] + self._cov_samples[i].T) / 2
            # equivalent to above
            # for j in range(Nm):
            #     innov = zs[j] - np.dot(self._H, self._state_samples[i])
            #     S = self._H @ self._cov_samples[i] @ self._H.T + (self._ext_samples[i] / 4 + self._R)
            #     S = (S + S.T) / 2
            #     K = self._cov_samples[i] @ self._H.T @ lg.inv(S)

            #     self._state_samples[i] += np.dot(K, innov)
            #     self._cov_samples[i] -= K @ S @ K.T
            #     self._cov_samples[i] = (self._cov_samples[i] + self._cov_samples[i].T) / 2
        self._weights /= self._weights.sum()

        # compute posterior extension, state and covariance
        self._state = 0
        self._cov = 0
        self._ext = 0
        for i in range(self._Ns):
            self._state += self._weights[i] * self._state_samples[i]
            self._cov += self._weights[i] * self._cov_samples[i]
            self._ext += self._weights[i] * self._ext_samples[i]
        self._cov = (self._cov + self._cov.T) / 2
        self._ext = (self._ext + self._ext.T) / 2

        # resample
        Neff = 1 / (self._weights**2).sum()
        if Neff <= self._Neff:
            idx = np.random.choice(np.arange(self._Ns), p=self._weights, size=self._Ns)
            self._ext_samples[:] = self._ext_samples[idx]
            self._state_samples[:] = self._state_samples[idx]
            self._cov_samples[:] = self._cov_samples[idx]
            self._weights[:] = 1 / self._Ns

        return self._state, self._cov, self._ext

    def distance(self, zs, **kwargs):
        return super().distance(zs, **kwargs)

    def likelihood(self, zs, **kwargs):
        return super().likelihood(zs, **kwargs)


class TurnRateEORBPFilter(EOFilterBase):
    '''
    Extended object Rao-Blackwellized particle filter with turning rate
    '''
    def __init__(self, F, H, D, R, Ns, Neff, df, T, omega_std=0, lamb=None):
        self._F = F.copy()
        self._H = H.copy()
        self._D = D.copy()
        self._R = R.copy()
        self._Ns = Ns
        self._Neff = Neff
        self._df = df
        self._omega_std = omega_std
        self._T = T
        self._lamb = lamb
        self._init = False

    def init(self, state, cov, df, extension, omega=0):
        self._state = state.copy()
        self._cov = cov.copy()
        self._ext = extension.copy()
        self._omega = omega

        # self._state_samples = st.multivariate_normal.rvs(state, cov, self._Ns)
        self._state_samples = np.full((self._Ns, state.shape[0]), state, dtype=float)
        self._cov_samples = np.full((self._Ns, cov.shape[0], cov.shape[1]), cov)
        self._ext_samples = st.wishart.rvs(df, extension / df, self._Ns)
        self._omega_samples = st.norm.rvs(omega, self._omega_std, self._Ns)
        self._weights = np.full(self._Ns, 1 / self._Ns, dtype=float)
        self._init = True

    def predict(self):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        # update samples
        for i in range(self._Ns):
            self._state_samples[i] = np.dot(self._F, self._state_samples[i])
            self._cov_samples[i] = self._F @ self._cov_samples[i] @ self._F.T + np.kron(self._ext_samples[i], self._D)
            A = rotate_matrix_deg(self._omega_samples[i] * self._T)
            self._ext_samples[i] = st.wishart.rvs(self._df, A @ self._ext_samples[i] @ A.T / self._df)
            self._omega_samples[i] = st.norm.rvs(self._omega_samples[i], self._omega_std)

        # compute prior extension, state and covariance
        self._state = 0
        self._cov = 0
        self._ext = 0
        self._omega = 0
        for i in range(self._Ns):
            self._state += self._weights[i] * self._state_samples[i]
            self._cov += self._weights[i] * self._cov_samples[i]
            self._ext += self._weights[i] * self._ext_samples[i]
            self._omega += self._weights[i] * self._omega_samples[i]
        self._cov = (self._cov + self._cov.T) / 2
        self._ext = (self._ext + self._ext.T) / 2

        return self._state, self._cov, self._ext

    def correct(self, zs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        Nm = len(zs)    # measurements number
        if self._lamb is None:
            lamb = Nm / ellip_volume(self._ext)        # empirical target density
        else:
            lamb = self._lamb

        H = np.kron(np.ones((Nm, 1)), self._H)      # expand dimension of H
        z_mean = np.mean(zs, axis=0)
        for i in range(self._Ns):
            # update weight
            V = ellip_volume(self._ext_samples[i])
            pmf = (lamb * V)**Nm * np.exp(-lamb * V) / sl.factorial(Nm)
            self._weights[i] *= pmf

            z_pred = np.dot(H, self._state_samples[i])
            R = self._ext_samples[i] / 4 + self._R
            # expanded dimension innovation covariance
            P = np.kron(np.eye(Nm), R) + H @ self._cov_samples[i] @ H.T
            if Nm <= 3:
                # calculate the inversion of P directly
                P_inv = lg.inv(P)
            else:
                # calculate the inversion of P using `matrix inversion lemma`
                A_inv = np.kron(np.eye(Nm), lg.inv(R))
                D = lg.inv(self._cov_samples[i])
                P_inv = A_inv - A_inv @ H @ lg.inv(D + H.T @ A_inv @ H) @ H.T @ A_inv
            # compute the likelihood of measurements, this is the key process
            z_vec = np.reshape(zs, -1)
            pdf = np.exp(-(z_vec - z_pred) @ P_inv @ (z_vec - z_pred) / 2) / np.sqrt(lg.det(2 * np.pi * P))
            self._weights[i] *= pdf
            # self._weights[i] *= st.multivariate_normal.pdf(z_vec, mean=z_pred, cov=P)     # too slow

            # update state and covariance using Kalman filter using mean measurement
            innov = z_mean - np.dot(self._H, self._state_samples[i])
            S = self._H @ self._cov_samples[i] @ self._H.T + R / Nm
            S = (S + S.T) / 2
            K = self._cov_samples[i] @ self._H.T @ lg.inv(S)

            self._state_samples[i] += np.dot(K, innov)
            self._cov_samples[i] -= K @ S @ K.T
            self._cov_samples[i] = (self._cov_samples[i] + self._cov_samples[i].T) / 2
            # equivalent to above
            # for j in range(Nm):
            #     innov = zs[j] - np.dot(self._H, self._state_samples[i])
            #     S = self._H @ self._cov_samples[i] @ self._H.T + (self._ext_samples[i] / 4 + self._R)
            #     S = (S + S.T) / 2
            #     K = self._cov_samples[i] @ self._H.T @ lg.inv(S)

            #     self._state_samples[i] += np.dot(K, innov)
            #     self._cov_samples[i] -= K @ S @ K.T
            #     self._cov_samples[i] = (self._cov_samples[i] + self._cov_samples[i].T) / 2
        self._weights /= self._weights.sum()

        # compute posterior extension, state and covariance
        self._state = 0
        self._cov = 0
        self._ext = 0
        self._omega = 0
        for i in range(self._Ns):
            self._state += self._weights[i] * self._state_samples[i]
            self._cov += self._weights[i] * self._cov_samples[i]
            self._ext += self._weights[i] * self._ext_samples[i]
            self._omega += self._weights[i] * self._omega_samples[i]
        self._cov = (self._cov + self._cov.T) / 2
        self._ext = (self._ext + self._ext.T) / 2

        # resample
        Neff = 1 / (self._weights**2).sum()
        if Neff <= self._Neff:
            idx = np.random.choice(np.arange(self._Ns), p=self._weights, size=self._Ns)
            self._ext_samples[:] = self._ext_samples[idx]
            self._omega_samples[:] = self._omega_samples[idx]
            self._state_samples[:] = self._state_samples[idx]
            self._cov_samples[:] = self._cov_samples[idx]
            self._weights[:] = 1 / self._Ns

        return self._state, self._cov, self._ext

    def distance(self, zs, **kwargs):
        return super().distance(zs, **kwargs)

    def likelihood(self, zs, **kwargs):
        return super().likelihood(zs, **kwargs)

    def omega(self):
        return self._omega
