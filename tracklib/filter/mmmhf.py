# -*- coding: utf-8 -*-
'''
Multiple model multiple hypothesis filter
'''
from __future__ import division, absolute_import, print_function


__all__ = ['MMMHFilter']

import numbers
import numpy as np
from collections.abc import Iterable
from .base import FilterBase
from tracklib.model import model_switch


class MMMMFQueue():
    def __init__(self, size):
        self._container = []
        self._max_size = size
        self._size = 0
    
    def put(self, v):
        if self._size < self._max_size:
            self._container.append(v)
            self._size += 1
        else:
            self._container.pop(0)
            self._container.append(v)
    
    def size(self):
        return self._size

    def copy(self):
        obj = MMMMFQueue(self._max_size)
        obj._container = self._container.copy()
        obj._size = self._size
        return obj

    def __eq__(self, value):
        return self._container == value._container


class MMMHFilter(FilterBase):
    '''
    Multiple model multiple hypothesis filter
    '''
    def __init__(self,
                 model_cls,
                 model_types,
                 init_args,
                 init_kwargs,
                 trans_mat=0.99,
                 history_probs=None,
                 depth=2,
                 keep=None,
                 pruning=1e-5,
                 switch_fcn=model_switch):
        super().__init__()

        self._models_n = len(model_cls)
        self._cls = model_cls
        self._models = [model_cls[i](*init_args[i], **init_kwargs[i]) for i in range(self._models_n)]
        self._idx = np.arange(self._models_n)
        self._types = model_types
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
        self._depth = depth
        self._cur_depth = 1
        if keep is None:
            self._keep = self._models_n**(self._depth)
        else:
            self._keep = keep
        self._hypos = []
        for i in range(self._models_n):
            h = MMMMFQueue(depth)
            h.put(i)
            self._hypos.append(h)
        self._pruning = pruning
        self._switch_fcn = switch_fcn
        self._is_first = True

    def __str__(self):
        msg = 'Multiple model multiple hypothesis filter'
        return msg

    def __len__(self):
        return self._models_n

    def __iter__(self):
        return ((self._models[i], self._probs[i]) for i in range(self._models_n))

    def __getitem__(self, n):
        if isinstance(n, (numbers.Integral, slice)):
            return self._models[n], self._probs[n]
        elif isinstance(n, Iterable):
            m = [self._models[i] for i in n]
            p = [self._probs[i] for i in n]
            return m, p
        else:
            raise TypeError("index must be an integer, slice or iterable, not '%s'" % n.__class__.__name__)

    def init(self, state, cov):
        if self._models_n == 0:
            raise RuntimeError('no models')

        for i in range(self._models_n):
            x = self._switch_fcn(state, self._types[0], self._cur_types[i])
            P = self._switch_fcn(cov, self._types[0], self._cur_types[i])
            self._models[i].init(x, P)
        self._state = state.copy()
        self._cov = cov.copy()
        self._init = True

    def reset(self, state, cov):
        pass

    def __update(self):
        state_org = [m.state for m in self._models]
        cov_org = [m.cov for m in self._models]
        types = [t for t in self._cur_types]

        xtmp = 0
        xi_list = []
        for i in range(self._models_n):
            xi = self._switch_fcn(state_org[i], types[i], self._types[0])
            xi_list.append(xi)
            xtmp += self._probs[i] * xi
        self._state = xtmp

        Ptmp = 0
        for i in range(self._models_n):
            xi = xi_list[i]
            Pi = self._switch_fcn(cov_org[i], types[i], self._types[0])
            err = xi - xtmp
            Ptmp += self._probs[i] * (Pi + np.outer(err, err))
        Ptmp = (Ptmp + Ptmp.T) / 2
        self._cov = Ptmp

    def __classify(self, lst):
        key = []
        index = []
        for i in range(len(lst)):
            if lst[i] not in key:
                key.append(lst[i])
                index.append([i])
            else:
                index[key.index(lst[i])].append(i)
        return key, index

    def __prune(self):
        max_idx = [i for i in range(self._models_n) if self._probs[i] >= self._pruning]
        max_idx = np.array(max_idx, dtype=int)

        if max_idx.size < self._models_n:
            self._models = [self._models[i] for i in max_idx]
            self._cur_types = [self._cur_types[i] for i in max_idx]
            self._probs = self._probs[max_idx]
            self._probs /= np.sum(self._probs)
            self._idx = self._idx[max_idx]
            self._hypos = [self._hypos[i] for i in max_idx]
            self._models_n = len(self._models)

    def __merge(self):
        if self._cur_depth == self._depth:
            max_idx = []
            _, index = self.__classify(self._hypos)
            for idx in index:
                max_idx.append(idx[np.argmax(self._probs[idx])])
            max_idx = np.array(max_idx, dtype=int)

            self._models = [self._models[i] for i in max_idx]
            self._cur_types = [self._cur_types[i] for i in max_idx]
            self._probs = self._probs[max_idx]
            self._probs /= np.sum(self._probs)
            self._idx = self._idx[max_idx]
            self._hypos = [self._hypos[i] for i in max_idx]
            self._models_n = len(self._models)

            if self._keep < self._models_n:
                max_idx = np.argsort(self._probs)[-self._keep:]
                self._models = [self._models[i] for i in max_idx]
                self._cur_types = [self._cur_types[i] for i in max_idx]
                self._probs = self._probs[max_idx]
                self._probs /= np.sum(self._probs)
                self._idx = self._idx[max_idx]
                self._hypos = [self._hypos[i] for i in max_idx]
                self._models_n = len(self._models)

    def __advance(self):
        models = []
        types = []
        probs = []
        idx = []
        hypos = []
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

                h = self._hypos[i].copy()
                h.put(j)
                hypos.append(h)
            types.extend(self._types)
        self._models = models
        self._cur_types = types
        self._probs = probs / np.sum(probs)
        self._idx = np.array(idx, dtype=int)
        self._hypos = hypos
        self._models_n = len(self._models)
        if self._cur_depth < self._depth:
            self._cur_depth += 1

    def predict(self, u=None, **kwargs):
        # print(kwargs['n'], self._cur_types[np.argmax(self._probs)], np.max(self._probs))
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        if self._is_first:    # do not prune and advance at the first prediction
            self._is_first = False
        else:
            self.__prune()
            self.__merge()
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

        # update the probabilities of history
        self._probs *= pdf
        self._probs /= np.sum(self._probs)

        self.__update()

        return self._state, self._cov

    def distance(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        d = 0
        for i in range(self._models_n):
            d += self._probs[i] * self._models[i].distance(z, **kwargs)

        return d

    def likelihood(self, z, **kwargs):
        if self._init == False:
            raise RuntimeError('filter must be initialized with init() before use')

        pdf = 0
        for i in range(self._models_n):
            pdf += self._probs[i] * self._models[i].likelihood(z, **kwargs)

        return pdf
