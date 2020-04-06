# -*- coding: utf-8 -*-
'''
Particle filter
'''
from __future__ import division, absolute_import, print_function

__all__ = []

import numpy as np
import scipy.linalg as lg
from .base import PFBase
from tracklib.utils import crndn


def SIRPFilter(PFBase):
    def __init__(self, f, h, Q, R, Ns=50):
        super().__init__()

        self._f = f
        self._h = h
        self._Q = Q
        self._R = R
        self._Ns = Ns

    def __str__(self):
        msg = 'SIR particle filter'
        return msg

    def __repr__(self):
        return self.__str__()

    def init(self, state, cov):
        pass

    def step(self, z):
        pass