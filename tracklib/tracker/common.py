# -*- coding: utf-8 -*-
'''
REFERENCES: B.-N. Vo, M. Mallick, Y. Bar-shalom, S. Coraluppi, R. Osborne, R. Mahler, et al., "Multitarget tracking", Wiley Encyclopedia of Electrical and Electronics Engineering, pp. 01-15, 2015.
'''
from __future__ import division, absolute_import, print_function


__all__ = ['HistoryLogic', 'ScoreLogic', 'Detection', 'TrackCounter']

import numbers
import numpy as np
import tracklib as tlb
from collections.abc import Iterable


class HistoryLogic():
    def __init__(self, confirm_M=2, confirm_N=3, delete_M=5, delete_N=5):
        self._c_M = confirm_M
        self._c_N= confirm_N
        self._d_M = delete_M
        self._d_N = delete_N
        max_N = max(confirm_N, delete_N)
        self._history = np.zeros(max_N, dtype=np.bool)
        self._history[0] = True

    def hit(self):
        self._history[1:] = self._history[:-1]
        self._history[0] = True

    def miss(self):
        self._history[1:] = self._history[:-1]
        self._history[0] = False

    def confirmed(self):
        return np.sum(self._history[:self._c_N] == True) >= self._c_M

    def detached(self, has_confirmed, age):
        if has_confirmed:
            if age > self._d_N:
                ret = np.sum(self._history[:self._d_N] == False) >= self._d_M
            else:
                ret = np.sum(self._history[:age] == False) >= self._d_M
        else:       # delete the track can not be confirmed
            left = self._c_N - age
            need = self._c_M - np.sum(self._history[:self._c_N] == True)
            ret = need > left
        return ret


class ScoreLogic():
    def __init__(self,
                 confirm_score=20,
                 delete_score=-5,
                 pd=0.9,
                 pfa=1e-6,
                 volume=1,
                 beta=1e-5):
        '''
        Parameters
        ----------
        confirm_score : number
            Confirmation threshold, specified as a positive scalar. If the logic score is above
            this threshold, then the track is confirmed.
        delete_score : number
            Deletion threshold, specified as a negative scalar. If the value of current Score minus
            max_score is more negative than the deletion threshold, then the track is deleted.
        volume : number
            Volume of sensor detection bin or of resolution cell. For example, a 2-D radar will have
            a sensor bin volume of (azimuth resolution in radians) * (range) * (range resolution).
        beta : number
            Rate of new targets in unit volume.
        pd : number
            Probability of detection.
        pfa : number
            Probability of false alarm for a detection bin.
        
        Note
        ----
        The clutter density = pfa / volume, so the default clutter density is equal to pfa
        '''
        self._c_score = confirm_score
        self._d_score = delete_score
        self._pd = pd
        self._pfa = pfa
        self._vol = volume
        self._beta = beta

        lamb = pfa / volume     # clutter density, rate of false target in unit volume
        self._score = np.log(pd * beta / lamb)
        self._max_score = self._score

    def hit(self, likelihood):
        self._score += np.log(self._vol * likelihood)
        self._score += np.log(self._pd / self._pfa)
        if self._score >= self._max_score:
            self._max_score = self._score

    def miss(self):
        self._score += np.log((1 - self._pd) / (1 - self._pfa))
        if self._score >= self._max_score:
            self._max_score = self._score

    def confirmed(self):
        return self._score >= self._c_score

    def detached(self):
        return (self._score - self._max_score) < self._d_score


class Detection():
    def __init__(self, meas, cov):
        if isinstance(meas, np.ndarray):
            self._meas = (meas,)
        elif isinstance(meas, Iterable):
            self._meas = tuple(meas)
        else:
            raise TypeError("error 'meas' type: '%s'" % meas.__class__.__name__)
        if isinstance(cov, np.ndarray):
            self._cov = (cov,)
        elif isinstance(cov, Iterable):
            self._cov = tuple(cov)
        else:
            raise TypeError("error 'cov' type: '%s'" % cov.__class__.__name__)
        if len(meas) != len(cov):
            raise ValueError("the lengths of 'meas' and 'cov' must be the same")
        self._len = len(meas)

    def __iter__(self):
        it = ((self._meas[i], self._cov[i]) for i in range(self._len))
        return it

    def __getitem__(self, n):
        if isinstance(n, numbers.Integral):
            return self._meas[n], self._cov[n]
        elif isinstance(n, Iterable):
            m = [self._meas[i] for i in n]
            c = [self._cov[i] for i in n]
            return m, c
        else:
            raise TypeError("index must be an integer, not '%s'" % n.__class__.__name__)

    def __len__(self):
        return self._len

    @property
    def meas(self):
        return self._meas

    @property
    def cov(self):
        return self._cov


class TrackCounter():
    def __init__(self):
        self._track_id = 0

    def increase(self):
        self._track_id += 1

    def count(self):
        return self._track_id