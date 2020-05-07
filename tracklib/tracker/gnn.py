# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function


__all__ = []

import numpy as np
import scipy.optimize as op
from .common import *


class GNNTrack():
    track_id = 0
    def __init__(self, filter, logic):
        self._ft = filter
        self._lgc = logic

        self._age = 1
        self._id = None

    def predict(self):
        self._ft.predict()

    def distance(self, meas):
        return self._filter.distance(meas)

    def assign(self, meas):
        self._filter.update(meas)
        if self._logic.type() == 'history':
            self._logic.hit()
        else:
            pass    # TODO other logic
        self._age += 1

        if self._logic.confirmed():
            self._id = GNNTrack.track_id
            GNNTrack.track_id += 1

    def coast(self):
        if self._logic.type() == 'history':
            self._logic.miss()
        else:
            pass    # TODO other logic
        self._age += 1

    def confirmed(self):
        return self._lgc.confirmed(self)

    def detached(self):
        return self._lgc.detached(self)

    def age(self):
        return self._age

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
    def __init__(self, filter_generator, filter_initializer, logic_maintainer, gate, assignment=op.linear_sum_assignment):
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
        meas_cov = detection.convariance()

        if len(self._tent_tracks) + len(self._conf_tracks) == 0:
            # form new tracks
            for meas in detection:
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
            cost_main = np.zeros(track_num, meas_num)
            virt_track = self._gate * np.eye(meas_num)
            virt_det = self._gate * np.eye(track_num)
            cost_zero = np.zeros((meas_num, track_num))
            for ti in range(track_num):
                for di in range(meas_num):
                    cost_main[ti, di] = track[ti].distance(detection[di])
            cost_matrix = np.block([[cost_main, virt_det], [virt_track, cost_zero]])

            # find best assignment, 
            track_idx, det_idx = self._assignment(cost_matrix)

            # update assigned tracks
            for i in range(len(track_idx)):
                tracks[track_idx[i]].assign(detection[det_idx[i]])

            # coast unassigned tracks
            left_track_idx = np.setdiff1d(np.arange(len(track)), track_idx)
            for i in left_track_idx:
                tracks[i].coast()

            # put confirmed track into confirmed tracks list
            self._conf_tracks.extend([t for t in self._tent_tracks if t.confirmed()])
            self._tent_tracks = [t for t in self._tent_tracks if not t.confirmed()]

            # deleta all detached tracks
            self._conf_tracks = [t for t in self._conf_tracks if not t.detached()]
            self._tent_tracks = [t for t in self._tent_tracks if not t.detached()]

            # form new tracks
            left_det_idx = np.setdiff1d(np.arange(len(detection)), det_idx)
            for i in left_det_idx:
                ft = self._ft_gen()
                self._ft_init(ft, detection[i], meas_cov)
                lgc = self._lgc_main()
                track = GNNTrack(ft, lgc)
                # add new track into tentative tracks list
                self._tent_tracks.append(track)

        self._len += 1
