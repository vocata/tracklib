# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as lg
from .kfbase import KFBase
from .kf import KFilter
from .ekf import EKFilter_1st, EKFilter_2ed
from ..utils import col

__all__ = ['MMFilter']


class MMFilter(KFBase):
    '''
    Hybird multiple model filter
    '''
    def __init__(self):
        super().__init__()
        self._model = []
        self._name = []
        self._prob = []
        self._weight_state = None
        self._maxprob_state = None
        self._model_n = 0

    # TODO
    # 完整的多模型描述
    def __str__(self):
        msg = 'Multiple model filter:\n'
        return msg

    def __repr__(self):
        return self.__str__()

    # TODO
    # 实现索引

    def init(self, state=None, cov=None):
        '''
        Initial filter

        Parameters
        ----------
        state : ndarray or list or None
            Initial prior state estimate
        cov : ndarray or list or None
            Initial error convariance matrix

        Note : if state or cov is None, the model has been initialized outside 

        Returns
        -------
            None
        '''
        if state is None and cov is None:
            pass
        else:
            if isinstance(state, np.ndarray) and isinstance(cov, np.ndarray):
                for model in self._model:
                    model.init(state, cov)
            elif isinstance(state, list) and isinstance(cov, list):
                for i in range(self._model_n):
                    if state[i] is None or cov[i] is None:
                        continue
                    self._model[i].init(state[i], cov[i])
            else:
                raise ValueError('state and cov must be a ndarray, list or None')
        self._len = 0
        self._stage = 0
        self._init = True

    # TODO
    # 实现相同索引判断
    def add_model(self, model, prob, name=''):
        '''
        Add new model

        Parameters
        ----------
        model : KFilter or EKFilter
            standard Kalman filter or extended Kalman filter
        probability : float
            model prior probability
        name : str
            model name(optional)

        Returns
        -------
            None
        '''
        if isinstance(model, (KFilter, EKFilter_1st, EKFilter_2ed)):
            self._model.append(model)
            self._prob.append(prob)
            self._name.append(name)
            self._model_n += 1

    def predict(self, u=None):
        assert (self._stage == 0)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        for model in self._model:
            model.predict(u)
        self._stage = 1

    def update(self, z):
        assert (self._stage == 1)
        if self._init == False:
            raise RuntimeError('The filter must be initialized with init() before use')

        pdf = []
        for model in self._model:
            model.update(z)
            r = model.innov
            S = model.innov_cov
            pdf.append((np.exp(-r.T @ lg.inv(S) @ r / 2) /
                        np.sqrt(lg.det(2 * np.pi * S))).item())

        # total probability
        total = np.dot(pdf, self._prob)
        # update all models' posterior probability
        for i in range(self._model_n):
            self._prob[i] = pdf[i] * self._prob[i] / total 

        # weighted state estimate
        states = np.array([model.post_state.reshape(-1) for model in self._model]).T
        self._weight_state = col(np.dot(states, self._prob))
        # max probability model state estimate
        self._maxprob_state = self._model[np.argmax(self._prob)].post_state

        self._len += 1
        self._stage = 0

    def step(self, z, u=None):
        assert (self._stage == 0)

        self.predict(u)
        self.update(z)
        
    @property
    def weight_state(self):
        return self._weight_state

    @property
    def maxprob_state(self):
        return self._maxprob_state

    @property
    def prob(self):
        return self._prob

    @property
    def prior_state(self):
        return [m.prior_state for m in self._model]

    @property
    def post_state(self):
        return [m.post_state for m in self._model]

    @property
    def prior_cov(self):
        return [m.prior_cov for m in self._model]

    @property
    def post_cov(self):
        return [m.post_cov for m in self._model]

    @property
    def innov(self):
        return [m.innov for m in self._model]

    @property
    def innov_cov(self):
        return [m.innov_cov for m in self._model]

    @property
    def gain(self):
        return [m.gain for m in self._model]
