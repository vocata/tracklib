# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function


__all__ = [
    'GNNTrack', 'GNNFilterGenerator', 'GNNFilterInitializer',
    'GNNLogicMaintainer', 'GNNTracker'
]

import numpy as np
from scipy.optimize import linear_sum_assignment
from .common import *


class GNNTrack():
    track_id = 0
    def __init__(self, filter, logic):
        self._ft = filter
        self._lgc = logic

        self._has_confirmed = False
        self._age = 1
        self._id = None

    def predict(self):
        self._ft.predict()

    def assign(self, meas, meas_cov):
        if meas_cov is None:
            self._ft.correct(meas)
        else:
            self._ft.correct(meas, R=meas_cov)

        if self._lgc.type() == 'history':
            self._lgc.hit()
        else:
            pass    # TODO other logic
        self._age += 1

        if not self._has_confirmed:
            if self._lgc.confirmed():
                self._id = GNNTrack.track_id
                GNNTrack.track_id += 1
                self._has_confirmed = True

    def coast(self):
        if self._lgc.type() == 'history':
            self._lgc.miss()
        else:
            pass    # TODO other logic
        self._age += 1

    def distance(self, meas, meas_cov):
        if meas_cov is None:
            d = self._ft.distance(meas)
        else:
            d = self._ft.distance(meas, R=meas_cov)
        return d

    def confirmed(self):
        return self._lgc.confirmed()

    def detached(self):
        return self._lgc.detached()

    @property
    def state(self):
        return self._ft.state

    @property
    def age(self):
        return self._age

    @property
    def id(self):
        return self._id


class GNNFilterGenerator():
    def __init__(self, filter_cls, *args, **kwargs):
        self._ft_cls = filter_cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        ft = self._ft_cls(*self._args, **self._kwargs)
        return ft


class GNNFilterInitializer():
    def __init__(self, init_fcn, *args, **kwargs):
        self._init_fcn = init_fcn
        self._args = args
        self._kwargs = kwargs

    def __call__(self, filter, meas, meas_cov):
        state, cov = self._init_fcn(meas, meas_cov, *self._args, **self._kwargs)
        filter.init(state, cov)


class GNNLogicMaintainer():
    def __init__(self, logic_cls, *args, **kwargs):
        self._lgc_cls = logic_cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        lgc = self._lgc_cls(*self._args, **self._kwargs)
        return lgc


class GNNTracker():
    def __init__(self, filter_generator, filter_initializer, logic_maintainer, gate, assignment=linear_sum_assignment):
        self._ft_gen = filter_generator
        self._ft_init = filter_initializer
        self._lgc_main = logic_maintainer
        self._assignment = assignment
        self._gate = gate

        self._tent_tracks = []
        self._conf_tracks = []

        self._len = 0

    def __len__(self):
        return self._len

    def history_tracks_num(self):
        return GNNTrack.track_id

    def current_tracks_num(self):
        return len(self._tent_tracks)

    def all_tracks(self):
        return self._conf_tracks

    def add_detection(self, detection):
        if len(self._tent_tracks) + len(self._conf_tracks) == 0:
            # form new tracks
            for meas, meas_cov in detection:
                ft = self._ft_gen()
                self._ft_init(ft, meas, meas_cov)
                lgc = self._lgc_main()
                track = GNNTrack(ft, lgc)
                # add new track into tentative tracks list
                self._tent_tracks.append(track)
        else:
            # get all tracks
            tracks = self._tent_tracks + self._conf_tracks

            # predict all tracks
            for track in tracks:
                track.predict()

            # form cost matrix
            track_num = len(tracks)
            meas_num = len(detection)
            cost_main = np.zeros((track_num, meas_num))
            virt_track = np.Inf * np.ones((meas_num, meas_num))
            np.fill_diagonal(virt_track, self._gate / 2)
            virt_det = np.Inf * np.ones((track_num, track_num))
            np.fill_diagonal(virt_det, self._gate / 2)
            cost_zero = np.zeros((meas_num, track_num))
            for ti in range(track_num):
                for di in range(meas_num):
                    meas, meas_cov = detection[di]
                    cost_main[ti, di] = tracks[ti].distance(meas, meas_cov)
            cost_matrix = np.block([[cost_main, virt_det], [virt_track, cost_zero]])

            # find best assignment,
            row_idx, col_idx = self._assignment(cost_matrix)
            agn_idx = [i for i in range(track_num) if col_idx[i] < meas_num]
            agn_tk = row_idx[agn_idx]
            unagn_tk = np.setdiff1d(np.arange(track_num), agn_tk)
            agn_meas = col_idx[agn_idx]
            unagn_meas = np.setdiff1d(np.arange(meas_num), agn_meas)

            # update assigned tracks
            for ti, mi in [(agn_tk[i], agn_meas[i]) for i in range(len(agn_idx))]:
                tracks[ti].assign(*detection[mi])

            # coast unassigned tracks
            for ti in unagn_tk:
                tracks[ti].coast()

            # put confirmed track into confirmed tracks list
            self._conf_tracks.extend([t for t in self._tent_tracks if t.confirmed()])
            self._tent_tracks = [t for t in self._tent_tracks if not t.confirmed()]

            # deleta all detached tracks
            self._conf_tracks = [t for t in self._conf_tracks if not t.detached()]
            self._tent_tracks = [t for t in self._tent_tracks if not t.detached()]

            # form new tracks
            for mi in unagn_meas:
                ft = self._ft_gen()
                meas, meas_cov = detection[mi]
                self._ft_init(ft, meas, meas_cov)
                lgc = self._lgc_main()
                track = GNNTrack(ft, lgc)
                # add new track into tentative tracks list
                self._tent_tracks.append(track)

        self._len += 1
