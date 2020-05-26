# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function


__all__ = [
    'JPDA_events', 'JPDA_clusters', 'JPDATrack', 'JPDAFilterGenerator',
    'JPDAFilterInitializer', 'JPDALogicMaintainer', 'JPDATracker'
]

import numpy as np
from .common import *


def JPDA_events(valid_mat):
    m, n = valid_mat.shape
    height = min(m, n - 1)  # or the height of tree

    meas_idx = [0] * height
    target_idx = [0] * height
    record = np.zeros((m, n - 1), dtype=bool)   # record the path

    tmp = np.zeros_like(valid_mat)
    tmp[:, 0] = 1
    event_mat = tmp.copy()
    splited_mat = [event_mat]

    j = 0
    level = 0  # equal to the length of meas_idx and target_idx
    while j < m:
        while level < height and j < m:
            # get next
            for i in range(1, n):
                if valid_mat[j, i] and i not in target_idx and not record[j, i - 1]:
                    record[j, i - 1] = True
                    meas_idx[level] = j
                    target_idx[level] = i
                    level += 1
                    # get next success, from the event matrix
                    event_mat = tmp.copy()
                    for L in range(level):
                        event_mat[meas_idx[L], target_idx[L]] = True
                        event_mat[meas_idx[L], 0] = False
                    splited_mat.append(event_mat)
                    break
            j += 1

        if level > 0:       # backtracking
            level -= 1
            if j - 1 != meas_idx[level]:
                record[meas_idx[level] + 1:, :] = False
            j = meas_idx[level]
            meas_idx[level] = 0
            target_idx[level] = 0

    return splited_mat


def JPDA_clusters(valid_mat):
    row_n, col_n = valid_mat.shape

    flag = np.zeros(col_n, dtype=bool)
    clusters = []
    for i in range(col_n):
        if flag[i]:
            continue

        tar = [i]
        meas_flag = valid_mat[:, i]
        changed = True
        while changed:
            changed = False
            for j in range(i + 1, col_n):
                if flag[j]:
                    continue
                if np.any(meas_flag & valid_mat[:, j]):
                    meas_flag |= valid_mat[:, j]
                    flag[j] = True
                    changed = True
                    tar.append(j)

        tar = sorted(tar)
        meas = [i for i in range(row_n) if meas_flag[i]]
        clusters.append((tar, meas))

    return clusters


class JPDATrack():
    def __init__(self, filter, logic, counter):
        self._ft = filter
        self._lgc = logic
        self._ctr = counter

        self._id = -1
        self._age = 1
        self._has_confirmed = False

    def _predict(self):
        self._ft.predict()

    def _assign(self, zs, probs, Rs):
        if isinstance(self._lgc, HistoryLogic):
            self._lgc.hit()
        self._ft.correct_JPDA(zs, probs, R=Rs)

        if not self._has_confirmed:
            if self._lgc.confirmed():
                self._id = self._ctr.count()
                self._ctr.increase()
                self._has_confirmed = True
        self._age += 1

    def _coast(self):
        if isinstance(self._lgc, HistoryLogic):
            self._lgc.miss()
        self._age += 1

    def _distance(self, z, R):
        return self._ft.distance(z, R=R)

    def _likelihood(self, z, R):
        return self._ft.likelihood(z, R=R)

    def _confirmed(self):
        if isinstance(self._lgc, HistoryLogic):
            return self._lgc.confirmed()

    def _detached(self):
        if isinstance(self._lgc, HistoryLogic):
            return self._lgc.detached(self._has_confirmed, self._age)

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


class JPDAFilterGenerator():
    def __init__(self, filter_cls, *args, **kwargs):
        if not hasattr(filter_cls, 'correct_JPDA'):
            raise TypeError(
                "'%s' cannot be used as underlying filter of JPDA" %
                filter_cls.__name__)
        self._ft_cls = filter_cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        ft = self._ft_cls(*self._args, **self._kwargs)
        return ft


class JPDAFilterInitializer():
    def __init__(self, init_fcn, *args, **kwargs):
        self._init_fcn = init_fcn
        self._args = args
        self._kwargs = kwargs

    def __call__(self, filter, z, R):
        state, cov = self._init_fcn(z, R, *self._args, **self._kwargs)
        filter.init(state, cov)

class JPDALogicMaintainer():
    def __init__(self, logic_cls, *args, **kwargs):
        self._lgc_cls = logic_cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        lgc = self._lgc_cls(*self._args, **self._kwargs)
        return lgc


class JPDATracker():
    def __init__(self,
                 filter_generator,
                 filter_initializer,
                 logic_maintainer,
                 gate=30,
                 pd=0.9,
                 pfa=1e-6,
                 volume=1,
                 init_threshold=0.1,
                 hit_miss_threshold=0.2):
        self._ft_gen = filter_generator
        self._ft_init = filter_initializer
        self._lgc_main = logic_maintainer
        self._gate = gate
        self._pd = pd
        self._pfa = pfa
        self._vol = volume
        self._den = pfa / volume     # clutter density, rate of false target in unit volume
        self._init_thres = init_threshold
        self._hit_miss_thres = hit_miss_threshold

        self._ctr = TrackCounter()
        self._tent_tracks = []
        self._conf_tracks = []

        self._len = 0

    def __del__(self):
        # reset the id counter when tracker is destroyed
        JPDATrack.track_id = 0

    def __len__(self):
        return self._len

    def history_tracks_num(self):
        return self._ctr.count()

    def current_tracks_num(self):
        return len(self._conf_tracks)

    def tracks(self):
        return self._conf_tracks

    def add_detection(self, detection):
        tracks = self._tent_tracks + self._conf_tracks
        if len(tracks) == 0:
            for z, R in detection:
                ft = self._ft_gen()
                self._ft_init(ft, z, R)
                lgc = self._lgc_main()
                track = JPDATrack(ft, lgc, self._ctr)
                self._tent_tracks.append(track)
        else:
            # predict all tracks
            for track in tracks:
                track._predict()

            # form the validation matrix, row means the target and column represents the measurement
            unasg_meas = []
            track_num = len(tracks)
            meas_num = len(detection)
            valid_mat = np.zeros((meas_num, track_num), dtype=bool)
            for mi in range(meas_num):
                all_zero = True
                for ti in range(track_num):
                    z, R = detection[mi]
                    if tracks[ti]._distance(z, R) < self._gate:
                        valid_mat[mi, ti] = True
                        all_zero = False
                if all_zero:
                    unasg_meas.append(mi)

            # divide into some clusters and coast the targets without measurement
            clusters = JPDA_clusters(valid_mat)
            for tar, meas in clusters:      # traverse all clusters
                tmp_mat = valid_mat[meas][:, tar]
                if tmp_mat.size > 0:
                    sub_valid_mat = np.ones((len(meas), len(tar) + 1), dtype=bool)
                    sub_valid_mat[:, 1:] = tmp_mat
                    event_list = JPDA_events(sub_valid_mat)

                    # compute the probabilites of association events in event_list
                    event_probs = []
                    for event in event_list:
                        item1 = item2 = 1
                        for j in range(event.shape[0]):
                            for i in range(1, event.shape[1]):
                                if event[j, i]:
                                    z, R = detection[meas[j]]
                                    pdf = tracks[tar[i - 1]]._likelihood(z, R)
                                    item1 *= (pdf / self._den)
                        for i in range(1, event.shape[1]):
                            for j in range(event.shape[0]):
                                if event[j, i]:
                                    item2 *= self._pd
                                    break
                            else:
                                item2 *= (1 - self._pd)
                        event_probs.append(item1 * item2)
                    event_probs = event_probs / np.sum(event_probs)

                    # compute the marginal association probabilities
                    beta = np.zeros((len(meas), len(tar)))
                    for j in range(beta.shape[0]):
                        for i in range(beta.shape[1]):
                            for event, prob in zip(event_list, event_probs):
                                beta[j, i] += prob * event[j, i + 1]

                    # update assigned tracks and coast the unassigned tracks
                    for i in range(beta.shape[1]):
                        if np.sum(beta[:, i]) < self._hit_miss_thres:
                            tracks[tar[i]]._coast()
                        else:
                            zs, Rs = detection[meas]
                            probs = beta[:, i]
                            tracks[tar[i]]._assign(zs, probs, Rs)

                    # find the measurements that association probability lower than init_threshold
                    # and initialize the tracks respectively later using these measurements
                    for j in range(beta.shape[0]):
                        if np.all(beta[j] < self._init_thres):
                            unasg_meas.append(meas[j])
                else:
                    tracks[tar[0]]._coast()     # the cluster of tracks without measurement only has one track

            # update confirmed and tentative list
            conf_tracks = []
            tent_tracks = []
            for t in self._conf_tracks:
                if not t._detached():
                    conf_tracks.append(t)
            for t in self._tent_tracks:
                if not t._detached():
                    if t._confirmed():
                        conf_tracks.append(t)
                    else:
                        tent_tracks.append(t)

            # form new tentative tracks using meas_init_flag
            for mi in unasg_meas:
                ft = self._ft_gen()
                z, R = detection[mi]
                self._ft_init(ft, z, R)
                lgc = self._lgc_main()
                track = JPDATrack(ft, lgc, self._ctr)
                tent_tracks.append(track)
            self._conf_tracks = conf_tracks
            self._tent_tracks = tent_tracks
