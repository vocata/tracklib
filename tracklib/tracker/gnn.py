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

        self._id = -1
        self._age = 1
        self._has_confirmed = False

    def _predict(self):
        self._ft.predict()

    def _assign(self, z, R):
        self._ft.correct(z, R=R)

        # update logic
        if isinstance(self._lgc, HistoryLogic):
            self._lgc.hit()
        else:
            # TODO other logic, such as score logic
            pass

        if not self._has_confirmed:
            if self._lgc.confirmed():
                self._id = GNNTrack.track_id
                GNNTrack.track_id += 1
                self._has_confirmed = True
        self._age += 1

    def _coast(self):
        # update logic
        if isinstance(self._lgc, HistoryLogic):
            self._lgc.miss()
        else:
            # TODO other logic, such as score logic
            pass
        self._age += 1

    def _distance(self, z, R):
        return self._ft.distance(z, R=R)

    def _confirmed(self):
        return self._lgc.confirmed()

    def _detached(self):
        return self._lgc.detached()

    def filter(self):
        return self._ft

    def logic(self):
        return self._lgc

    @property
    def state(self):
        return self._ft.state

    @property
    def cov(self):
        return self._ft.cov

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

    def __call__(self, filter, z, R):
        state, cov = self._init_fcn(z, R, *self._args, **self._kwargs)
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
    def __init__(self, filter_generator, filter_initializer, logic_maintainer, threshold, assignment=linear_sum_assignment):
        self._ft_gen = filter_generator
        self._ft_init = filter_initializer
        self._lgc_main = logic_maintainer
        self._asg_fcn = assignment
        self._thres = threshold

        self._tent_tracks = []
        self._conf_tracks = []

        self._len = 0

    def __len__(self):
        return self._len

    def history_tracks_num(self):
        return GNNTrack.track_id

    def current_tracks_num(self):
        return len(self._conf_tracks)

    def tracks(self):
        return self._conf_tracks

    def add_detection(self, detection):
        if len(self._tent_tracks) + len(self._conf_tracks) == 0:
            for z, R in detection:
                # generate new filter
                ft = self._ft_gen()
                # initialize filter
                self._ft_init(ft, z, R)
                # obtain a new logic maintainer
                lgc = self._lgc_main()
                # form a new tentative track
                track = GNNTrack(ft, lgc)
                # add new track into tentative tracks list
                self._tent_tracks.append(track)
        else:
            # get all tracks
            tracks = self._tent_tracks + self._conf_tracks

            # predict all tracks
            for track in tracks:
                track._predict()

            # form cost matrix
            track_num = len(tracks)
            meas_num = len(detection)
            cost_main = np.zeros((track_num, meas_num))
            virt_track = np.full((meas_num, meas_num), np.Inf)
            np.fill_diagonal(virt_track, self._thres / 2)
            virt_det = np.full((track_num, track_num), np.Inf)
            np.fill_diagonal(virt_det, self._thres / 2)
            cost_zero = np.zeros((meas_num, track_num))
            for ti in range(track_num):
                for di in range(meas_num):
                    z, R = detection[di]
                    cost_main[ti, di] = tracks[ti]._distance(z, R)
            cost_matrix = np.block([[cost_main, virt_det], [virt_track, cost_zero]])

            # find best assignment,
            row_idx, col_idx = self._asg_fcn(cost_matrix)
            asg_idx = [i for i in range(track_num) if col_idx[i] < meas_num]
            asg_tk = row_idx[asg_idx]
            unasg_tk = np.setdiff1d(np.arange(track_num), asg_tk)
            asg_meas = col_idx[asg_idx]
            unasg_meas = np.setdiff1d(np.arange(meas_num), asg_meas)

            # update assigned tracks
            for ti, mi in [(asg_tk[i], asg_meas[i]) for i in range(len(asg_idx))]:
                tracks[ti]._assign(*detection[mi])

            # coast unassigned tracks
            for ti in unasg_tk:
                tracks[ti]._coast()

            # update confirmed list and tentative list
            self._conf_tracks = [t for t in tracks if t._confirmed() and not t._detached()]
            self._tent_tracks = [t for t in tracks if not t._confirmed() and not t._detached()]

            # form new tentative tracks using unassigned measurements
            for mi in unasg_meas:
                ft = self._ft_gen()
                z, R = detection[mi]
                self._ft_init(ft, z, R)
                lgc = self._lgc_main()
                track = GNNTrack(ft, lgc)
                self._tent_tracks.append(track)

        self._len += 1
