# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function


__all__ = []

import numpy as np


class GNNFilterGenerator():
    def __init__(self, filter, *args, **kwargs):
        self._filter = filter
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        filter = self._filter(*self._args, **self._kwargs)
        return filter


class GNNIntializer():
    def __init__(self, initializer, *args, **kwargs):
        self._initializer = initializer
        self._args = args
        self._kwargs = kwargs

    def init(self, filter):
        state, cov = self._initializer(*self._args, **self._kwargs)
        filter.init(state, cov)


class GNNTracker():
    def __init__(self, filter_generator):
        self._filter_generator = filter_generator
        self._tentative_tracks = []
        self._confirmed_tracks = []
        self._unassigned_points = []
        self._id_history = 0
        self._len = 0

    def add_detection(self, detections):
        if len(self._tentative_tracks) + len(self._confirmed_tracks) == 0:
            pass
        else:
            pass
        self._len += 1

