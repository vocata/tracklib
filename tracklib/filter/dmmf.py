# -*- coding: utf-8 -*-
'''
Dynamic multiple model filter

REFERENCE:
[1]. Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation: Theory, Algorithms and Software," New York: Wiley, 2001
'''
from __future__ import division, absolute_import, print_function


__all__ = ['IMMFilter']

import numbers
import numpy as np
import scipy.linalg as lg
from collections.abc import Iterable
from .base import FilterBase
from tracklib.model import model_switch


class IMMFilter(FilterBase):
    '''
    Interacting multiple model filter
    '''
    def __init__(self, model_cls, model_types, init_args, init_kwargs, trans_mat=0.999, model_probs=None, switch_fcn=model_switch):
        super().__init__()

        self._models_n = len(model_cls)
        self._models = [model_cls[i](*init_args[i], **init_kwargs[i]) for i in range(self._models_n)]
        self._types = model_types
        if model_probs is None:
            self._probs = np.full(self._models_n, 1 / self._models_n, dtype=float)
        else:
            self._probs = model_probs
        if self._models_n == 1:
            self._trans_mat = np.eye(1)
        elif isinstance(trans_mat, numbers.Number):
            other_probs = (1 - trans_mat) / (self._models_n - 1)
            self._trans_mat = np.full((self._models_n, self._models_n), other_probs)
            np.fill_diagonal(self._trans_mat, trans_mat)
        else:
            self._trans_mat = trans_mat
        self._switch_fcn = switch_fcn

    def __str__(self):
        msg = 'Interacting multiple model filter:\n{\n  '
        if self._models_n < 10:
            sub = ['{}: model: {}, probability: {}'.format(i, self._models[i], self._probs[i]) for i in range(self._models_n)]
            sub = '\n  '.join(sub)
        else:
            sub = ['{}: model: {}, probability: {}'.format(i, self._models[i], self._probs[i]) for i in range(3)]
            sub.append('...')
            sub.extend(['{}: model: {}, probability: {}'.format(i, self._models[i], self._probs[i]) for i in range(self._models_n - 3, self._models_n)])
            sub = '\n  '.join(sub)
        msg += sub
        msg += '\n}'
        return msg

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return ((self._models[i], self._probs[i]) for i in range(self._models_n))

    def __getitem__(self, n):
        if isinstance(n, numbers.Integral):
            return self._models[n], self._probs[n]
        elif isinstance(n, Iterable):
            m = [self._models[i] for i in n]
            p = [self._probs[i] for i in n]
            return m, p
        else:
            raise TypeError('index can not be the type: `%s`' % n.__class__.__name__)

    def __update(self):
        state_org = [self._models[i].state for i in range(self._models_n)]
        cov_org = [self._models[i].cov for i in range(self._models_n)]
        types = [self._types[i] for i in range(self._models_n)]

        xtmp = 0
        for i in range(self._models_n):
            xi = self._switch_fcn(state_org[i], types[i], types[0])
            xtmp += self._probs[i] * xi
        self._state = xtmp

        Ptmp = 0
        for i in range(self._models_n):
            xi = self._switch_fcn(state_org[i], types[i], types[0])
            pi = self._switch_fcn(cov_org[i], types[i], types[0])
            err = xi - xtmp
            Ptmp += self._probs[i] * (pi + np.outer(err, err))
        Ptmp = (Ptmp + Ptmp.T) / 2
        self._cov = Ptmp

    def init(self, state, cov):
        '''
        Initial filter

        Parameters
        ----------
        state : ndarray
            Initial prior state estimate
        cov : ndarray
            Initial error convariance matrix

        Returns
        -------
            None
        '''
        if self._models_n == 0:
            raise RuntimeError('models must be added before calling init')

        for i in range(self._models_n):
            x = self._switch_fcn(state, self._types[0], self._types[i])
            P = self._switch_fcn(cov, self._types[0], self._types[i])
            self._models[i].init(x, P)
        self._state = state.copy()
        self._cov = cov.copy()
        self._init = True

    def reset(self, state, cov):
        if self._models_n == 0:
            raise AttributeError("AttributeError: can't set attribute")

        for i in range(self._models_n):
            xi = self._switch_fcn(state, self._types[0], self._types[i])
            Pi = self._switch_fcn(cov, self._types[0], self._types[i])
            self._models[i].reset(xi, Pi)

    def predict(self, u=None, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        # mixing/interaction, the difference from the GPB1 and GPB2 is that merging
        # process (called mixing here) is carried out at the beginning of cycle.
        mixing_probs = self._trans_mat * self._probs
        # prior model probability P(M(k)|Z^(k-1))
        self._probs = np.sum(mixing_probs, axis=1)
        # mixing probability P(M(k-1)|M(k),Z^(k-1))
        mixing_probs /= self._probs.reshape(-1, 1)
        # mixing
        state_org = [self._models[i].state for i in range(self._models_n)]
        cov_org = [self._models[i].cov for i in range(self._models_n)]
        types = [self._types[i] for i in range(self._models_n)]

        mixed_state = []
        for i in range(self._models_n):
            xi = 0
            for j in range(self._models_n):
                xj = self._switch_fcn(state_org[j], types[j], types[i])
                xi += mixing_probs[i, j] * xj
            mixed_state.append(xi)
        for i in range(self._models_n):
            Pi = 0
            xi = mixed_state[i]
            for j in range(self._models_n):
                xj = self._switch_fcn(state_org[j], types[j], types[i])
                Pj = self._switch_fcn(cov_org[j], types[j], types[i])
                err = xj - xi
                Pi += mixing_probs[i, j] * (Pj + np.outer(err, err))
            Pi = (Pi + Pi.T) / 2
            self._models[i].reset(xi, Pi)

        for i in range(self._models_n):
            self._models[i].predict(u, **kwargs)
        # update prior state and covariance
        self.__update()

        return self._state, self._cov

    def correct(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        pdf = np.zeros(self._models_n)
        for i in range(self._models_n):
            pdf[i] = self._models[i].likelihood(z, **kwargs)
            self._models[i].correct(z, **kwargs)
        # posterior model probability P(M(k)|Z^k)
        self._probs *= pdf
        self._probs /= np.sum(self._probs)
        # update posterior state and covariance
        self.__update()

        return self._state, self._cov

    def correct_JPDA(self, zs, probs, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        z_len = len(zs)
        kwargs_list = [{}] * z_len
        # group the keyword arguments
        for key, value in kwargs.items():
            for vi in range(z_len):
                kwargs_list[vi][key] = value[vi]

        pdfs = np.zeros((self._models_n, z_len))
        for i in range(self._models_n):
            for j in range(z_len):
                pdfs[i, j] = self._models[i].likelihood(zs[j], **kwargs_list[j])
            self._models[i].correct_JPDA(zs, probs, **kwargs)
        
        # posterior model probability P(M(k)|Z^k)
        pdf = np.dot(pdfs, probs) + (1 - np.sum(probs))     # ??
        self._probs *= pdf
        self._probs /= np.sum(self._probs)
        # update posterior state and covariance
        self.__update()

        return self._state, self._cov

    def distance(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        d = 0
        for i in range(self._models_n):
            d += self._probs[i] * self._models[i].distance(z, **kwargs)

        return d

    def likelihood(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('the filter must be initialized with init() before use')

        pdf = 0
        for i in range(self._models_n):
            pdf += self._probs[i] * self._models[i].likelihood(z, **kwargs)

        return pdf

    def models(self):
        return self._models

    def probs(self):
        return self._probs

    def trans_mat(self):
        return self._trans_mat
