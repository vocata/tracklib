# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function


__all__ = []

import numpy as np
from .common import *


class GNNFilterGenerator():
    def __init__(self, filter_cls, *args, **kwargs):
        self._filter_cls = filter_cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        filter = self._filter_cls(*self._args, **self._kwargs)
        return filter


class GNNFilterInitializer():
    def __init__(self, init_fcn, *args, **kwargs):
        self._init_fcn = init_fcn
        self._args = args
        self._kwargs = kwargs

    def __call__(self, filter, target, covariance):
        state, cov = self._init_fcn(target, covariance, *self._args, **self._kwargs)
        filter.init(state, cov)


class GNNLogicMaintainer():
    def __init__(self, logic_cls, *args, **kwargs):
        self._logic_cls = logic_cls
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self):
        logic = self._logic_cls(*self._args, **self._kwargs)
        return logic


class GNNTracker():
    def __init__(self, filter_generator, filter_initializer, logic_maintainer, gate):
        self._filter_generator = filter_generator
        self._filter_initializer = filter_initializer
        self._logic_maintainer = logic_maintainer
        self._gate = gate

        self._tentative_tracks = []
        self._confirmed_tracks = []

        self._len = 0

    def __len__(self):
        return self._len
    

    def add_detection(self, detection):
        if len(self._tentative_tracks) + len(self._confirmed_tracks) == 0:
            cov = detection.convariance()
            for target in detection:
                filter = self._filter_generator()
                self._initializer(filter, target, cov)
                logic = self._logic_maintainer()
                track = Track(filter, logic)
                self._tentative_tracks.append(track)
        else:
            pass
        self._len += 1
