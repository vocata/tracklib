# -*- coding: utf-8 -*-

import abc
import numpy as np

__all__ = []


class KFBase(object, metaclass=abc.ABCMeta):
    def __init__(self, x_dim=1, z_dim=1, debug=False):
        '''
        x_dim: state vector dimension
        z_dim: measurement vector dimension
        debug: debug mode, record all process variables 

        default is scalar kalman filter
        '''

        self._x_dim = x_dim
        self._z_dim = z_dim
        x_shape = (x_dim, 1)
        z_shape = (z_dim, 1)
        P_shape = (x_dim, x_dim)
        inP_shape = (z_dim, z_dim)
        K_shape = (x_dim, z_dim)

        self._x_pred = np.zeros(x_shape)
        self._P_pred = np.zeros(P_shape)
        self._x_up = np.zeros(x_shape)
        self._P_up = np.zeros(P_shape)
        self._innov = np.zeros(z_shape)
        self._inP = np.zeros(inP_shape)
        self._K = np.zeros(K_shape)

        self._x_init = np.zeros(x_shape)
        self._P_init = np.zeros(P_shape)

        self._len = 0
        self._stage = 0
        self._debug = debug

        if self._debug:
            self.__arr_init()

    def __len__(self):
        return self._len

    def __call__(self):
        return self.predict_info() + self.update_info()

    def __arr_init(self):
        self._x_pred_arr = []
        self._P_pred_arr = []
        self._x_up_arr = []
        self._P_up_arr = []
        self._innov_arr = []
        self._inP_arr = []
        self._K_arr = []

    def init(self, x_init, P_init):
        self._x_init = x_init
        self._P_init = P_init
        self._x_up = x_init
        self._P_up = P_init
        self._len = 0
        self._stage = 0
        if self._debug:
            self.__arr_init()

    @abc.abstractmethod
    def predict(self, *args, **kw):
        pass

    @abc.abstractmethod
    def update(self, *args, **kw):
        pass

    @abc.abstractmethod
    def step(self, *args, **kw):
        pass

    def init_info(self):
        return self._x_init, self._P_init

    def predict_info(self):
        return self._x_pred, self._P_pred

    def update_info(self):
        ret = (self._x_up, self._P_up, self._innov, self._inP, self._K)
        return ret

    def debug_result(self):
        assert (self._debug == 1)

        ret = (self._x_pred_arr, self._P_pred_arr, self._x_up_arr,
               self._P_up_arr, self._innov_arr, self._inP_arr, self._K_arr)
        ret = tuple(map(np.array, ret))
        return ret
