# -*- coding: utf-8 -*-
'''
Hypothesis multiple model filter
'''
from __future__ import division, absolute_import, print_function


__all__ = ['HMMFilter']

import numbers
import numpy as np
import scipy.linalg as lg
from .base import FilterBase
from tracklib.model import model_switch


class HMMFilter(FilterBase):
    '''
    Hypothesis multiple model filter
    '''
    def __init__(self,
                 model_cls,
                 model_types,
                 init_args,
                 init_kwargs,
                 trans_mat=0.99,
                 history_probs=None,
                 depth=3,
                 left=3,
                 switch_fcn=model_switch):
        super().__init__()

        self._models_n = len(model_cls)
        self._cls = model_cls
        self._models = [model_cls[i](*init_args[i], **init_kwargs[i]) for i in range(self._models_n)]
        self._idx = np.arange(self._models_n)
        self._llds = np.zeros(self._models_n)
        self._types = model_types
        self._type = model_types[0]
        self._cur_types = model_types
        self._args = init_args
        self._kwargs = init_kwargs
        if self._models_n == 1:
            self._trans_mat = np.eye(1)
        elif isinstance(trans_mat, numbers.Number):
            other_probs = (1 - trans_mat) / (self._models_n - 1)
            self._trans_mat = np.full((self._models_n, self._models_n), other_probs, dtype=float)
            np.fill_diagonal(self._trans_mat, trans_mat)
        else:
            self._trans_mat = trans_mat
        if history_probs is None:
            self._probs = np.full(self._models_n, 1 / self._models_n, dtype=float)
        else:
            self._probs = history_probs
        self._max_depth = depth
        self._depth = 0
        self._left = left
        self._switch_fcn = switch_fcn

    def __update(self):
        state_org = [m.state for m in self._models]
        cov_org = [m.cov for m in self._models]
        types = [t for t in self._cur_types]

        xtmp = 0
        xi_list = []
        for i in range(self._models_n):
            xi = self._switch_fcn(state_org[i], types[i], self._type)
            xi_list.append(xi)
            xtmp += self._probs[i] * xi
        self._state = xtmp

        Ptmp = 0
        for i in range(self._models_n):
            xi = xi_list[i]
            Pi = self._switch_fcn(cov_org[i], types[i], self._type)
            err = xi - xtmp
            Ptmp += self._probs[i] * (Pi + np.outer(err, err))
        Ptmp = (Ptmp + Ptmp.T) / 2
        self._cov = Ptmp

    def init(self, state, cov):
        if self._models_n == 0:
            raise RuntimeError('no models')

        for i in range(self._models_n):
            x = self._switch_fcn(state, self._type, self._cur_types[i])
            P = self._switch_fcn(cov, self._type, self._cur_types[i])
            self._models[i].init(x, P)
        self._state = state.copy()
        self._cov = cov.copy()
        self._init = True

    def reset(self, state, cov):
        pass

    def __pruning(self):
        if self._depth >= self._max_depth:
            max_idx = np.argsort(self._llds)[-self._left:]
            self._llds = self._llds[max_idx]
            self._llds -= np.min(self._llds)        # avoid overflow
            self._models = [self._models[i] for i in max_idx]
            self._cur_types = [self._cur_types[i] for i in max_idx]
            self._probs = self._probs[max_idx]
            self._probs /= np.sum(self._probs)
            self._idx = max_idx % len(self._cls)
            self._models_n = len(self._models)
            self._depth = 0

    def __advance(self):
        models = []
        types = []
        probs = []
        idx = []
        for i in range(self._models_n):
            state, cov = self._models[i].state, self._models[i].cov
            for j in range(len(self._cls)):
                x = self._switch_fcn(state, self._cur_types[i], self._types[j])
                P = self._switch_fcn(cov, self._cur_types[i], self._types[j])
                model = self._cls[j](*self._args[j], **self._kwargs[j])
                model.init(x, P)
                models.append(model)

                prob = self._trans_mat[j, self._idx[i]] * self._probs[i]
                probs.append(prob)

                idx.append(j)
            types.extend(self._types)
        self._models = models
        self._cur_types = types
        self._probs = probs / np.sum(probs)
        self._idx = np.array(idx, dtype=int)
        self._models_n = len(self._models)
        self._depth += 1

    def predict(self, u=None, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        self.__pruning()
        self.__advance()

        for m in self._models:
            m.predict(u, **kwargs)

        self.__update()

        return self._state, self._cov

    def correct(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        pdf = np.zeros(self._models_n)
        for i in range(self._models_n):
            pdf[i] = self._models[i].likelihood(z, **kwargs)
            self._models[i].correct(z, **kwargs)
        self._llds = np.log(np.reshape(pdf, (len(self._llds), -1))) + np.reshape(self._llds, (-1, 1))
        self._llds = np.reshape(self._llds, -1)

        self._probs *= pdf
        self._probs /= np.sum(self._probs)

        self.__update()

        return self._state, self._cov

    def distance(self, z, **kwargs):
        pass

    def likelihood(self, z, **kwargs):
        pass
