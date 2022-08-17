# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.math cimport isnan, NAN, INFINITY, fabs, sqrt

from libcpp.vector cimport vector

from cpython cimport array
import array

np.import_array()

"""
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef enum DIST_NORM: DIST_NORM_L1, DIST_NORM_L2, DIST_NORM_L2SQ
ctypedef enum DIST_REDUCE: DIST_REDUCE_MEDIAN, DIST_REDUCE_MEAN, DIST_REDUCE_LAST
"""

cdef extern from "<algorithm>" namespace "std" nogil:
    void nth_element[Iter](Iter first, Iter nth, Iter last) except +
    # void nth_element[ExecutionPolicy, Iter](ExecutionPolicy&& policy, Iter first, Iter nth, Iter last) except +
    # void nth_element[Iter, Compare](Iter first, Iter nth, Iter last, Compare comp) except +
    # void nth_element[ExecutionPolicy, Iter, Compare](ExecutionPolicy&& policy, Iter first, Iter nth, Iter last, Compare comp) except +

cdef array.array float_array_type = array.array('f', [])

cdef int _imax(int a, int b):
    return a if a > b else b

cdef class TrajectoryC:
    def __init__(self, points, cur_time, dist_norm, dist_reduce, inv_point, dist_func=DIST_FUNC_C) -> None:
        # self.points = np.array(points)
        for point in points:
            self.append(point)

        self.start_time = cur_time
        self.dist_norm = dist_norm
        self.dist_reduce = dist_reduce
        self.dist_func = dist_func
        self.inv_point = inv_point

    cdef np.ndarray[DTYPE_t, ndim=2] get_points_c(self):
        return np.asarray(<float [:self.points.size()]>self.points.data()).reshape(self.num_points(), 2)

    @staticmethod
    def from_trajectory(traj, *args, **kwargs):
        return TrajectoryC(traj.points, traj.start_time, traj.dist_norm, traj.dist_reduce, traj.inv_point, *args, **kwargs)

    def get_points(self):
        return self.get_points_c()

    """
    def get_points_np(self):
        x = np.asarray(<float [:self.points.size()]>self.points.data()).reshape(self.points.size() // 2, 2)
        return x
    """

    """
    def __getitem__(self, i):
        return self.points[i]
    """

    def __setitem__(self, i, val):
        if i < 0:
            i += self.num_points()

        self.points[2*i] = val[0]
        self.points[2*i+1] = val[1]

    def append(self, x):
        # self.points.append(x)
        # self.points = np.append(self.points, x[None], 0)
        self.points.push_back(x[0])
        self.points.push_back(x[1])

    cdef float _dist_norm(self, np.ndarray[DTYPE_t, ndim=2] points1, np.ndarray[DTYPE_t, ndim=2] points2):
        cdef np.ndarray[DTYPE_t, ndim=2] diffs
        cdef np.ndarray[DTYPE_t, ndim=1] dists
        cdef float res

        if self.dist_reduce == DIST_REDUCE_LAST:
            points1 = points1[points1.shape[0]-1:]
            points2 = points2[points2.shape[0]-1:]

        diffs = points2 - points1

        # TODO: if not overlapping, all entries will be nan in a subset of the array -> set distance to inf
        # otherwise ignore nan values

        if self.dist_norm == DIST_NORM_L2SQ:
            dists = np.sum(diffs * diffs, 1)
        elif self.dist_norm == DIST_NORM_L2:
            dists = np.sqrt(np.sum(diffs * diffs, 1))
        elif self.dist_norm == DIST_NORM_L1:
            dists = np.sum(np.abs(diffs), 1)

        if self.dist_reduce == DIST_REDUCE_MEDIAN:
            res = np.nanmedian(dists)
        elif self.dist_reduce == DIST_REDUCE_MEAN:
            res = np.nanmean(dists)
        elif self.dist_reduce == DIST_REDUCE_LAST:
            # assert dists.shape[1] == 1
            res = dists[dists.shape[0]-1]

        if isnan(res):
            res = np.inf

        return res

    cpdef float distance_to(self, TrajectoryC traj):
        if self.dist_func == DIST_FUNC_NP:
            return self.distance_to_np(traj)
        elif self.dist_func == DIST_FUNC_C:
            return self.distance_to_c(traj)
        else:
            return -1.0

    cdef float distance_to_np(self, TrajectoryC traj):
        cdef int t = np.maximum(self.start_time, traj.start_time)

        cdef np.ndarray[DTYPE_t, ndim=2] p1 = self.get_points_c()[t - self.start_time:]
        cdef np.ndarray[DTYPE_t, ndim=2] p2 = traj.get_points_c()[t - traj.start_time:]

        # assert len(p1) == len(p2)

        return self._dist_norm(p1, p2)

    cdef float distance_to_c(self, TrajectoryC traj):
        # all traj extend to current timestamp, only starttime differs
        cdef int t = _imax(self.start_time, traj.start_time) if self.dist_reduce != DIST_REDUCE_LAST else (self.start_time + self.num_points() - 1)
        cdef int length = self.num_points() - (t - self.start_time)

        # move pointer ahead in time to align
        cdef float* points1 = self.points.data() + 2 * (t - self.start_time)
        cdef float* points2 = traj.points.data() + 2 * (t - traj.start_time)

        cdef float[:] dists = array.clone(float_array_type, length, zero=False)  # max length if no number is nan
        cdef float* dists_ptr = <float*>&dists[0]
        cdef float p1_x, p1_y, p2_x, p2_y, dist_x, dist_y, dist, dist_res
        cdef int i, n, dist_count

        dist_res = 0
        dist_count = 0
        # dists.resize(length)  # index and than 2*i

        for i in range(0, 2 * length, 2):
            p1_x = points1[i+0]
            p1_y = points1[i+1]
            p2_x = points2[i+0]
            p2_y = points2[i+1]

            if not (isnan(p1_x) or isnan(p2_x)):
                dist_x = p2_x - p1_x
                dist_y = p2_y - p1_y

                if self.dist_norm == DIST_NORM_L2SQ:
                    dist = dist_x * dist_x + dist_y * dist_y
                elif self.dist_norm == DIST_NORM_L2:
                    dist = sqrt(dist_x * dist_x + dist_y * dist_y)
                elif self.dist_norm == DIST_NORM_L1:
                    dist = fabs(dist_x) + fabs(dist_y)

                dists[dist_count] = dist

                """
                if self.dist_reduce == DIST_REDUCE_MEDIAN:
                    res = np.nanmedian(dists)
                elif self.dist_reduce == DIST_REDUCE_MEAN:
                    res = np.nanmean(dists)
                elif self.dist_reduce == DIST_REDUCE_LAST:
                    # assert dists.shape[1] == 1
                    res = dists[dists.shape[0]-1]
                """

                dist_count += 1

        if self.dist_reduce == DIST_REDUCE_MEDIAN:
            # n = (dist_count-1)/2
            n = dist_count/2
            nth_element(dists_ptr, dists_ptr+n, dists_ptr+dist_count)
            dist_res = dists_ptr[n]
        elif self.dist_reduce == DIST_REDUCE_MEAN:
            for i in range(dist_count):
                dist_res += dists[i]
            dist_res /= dist_count
        elif self.dist_reduce == DIST_REDUCE_LAST:
            if dist_count > 0:
                dist_res = dists[0]

        if dist_count == 0:
            return INFINITY
        else:
            return dist_res

        # assert len(p1) == len(p2)

        # return self._dist_norm(p1, p2)

    """
    def distances_to(self, trajs):
        t = max(self.start_time, max(traj.start_time for traj in trajs))

        p1 = self.points[t - self.start_time:]
        p2 = [traj.points[t - self.start_time:] for traj in trajs]

        assert all(len(p1) == len(p) for p in p2)

        return self._dist_norm(p1, p2)
    """

    def align(self, start, end=None, points_only=True):
        assert points_only

        points = self.get_points()

        if start >= self.start_time:
            # cut
            p = points[start - self.start_time:]
        else:
            # pad
            p = np.concatenate([
                np.repeat(self.inv_point[None], self.start_time - start, 0),
                points
            ], 0)

        if end is not None:
            assert len(p) >= end - start
            p = p[:end - start]

        return p

    cdef int num_points(self):
        return self.points.size() // 2

    cdef int get_start_time(self):
        return self.start_time

    """
    def __len__(self):
        return len(self.points)
    """

    cpdef DIST_NORM get_dist_norm_type(self):
        return self.dist_norm

    cpdef DIST_REDUCE get_dist_reduce_type(self):
        return self.dist_reduce

    """
    @staticmethod
    def mean(trajs):
        t = min(traj.get_start_time() for traj in trajs)

        points = [traj.align(t) for traj in trajs]
        mean_traj = np.nanmean(points, 0)

        return TrajectoryC(mean_traj, t)
    """

    @staticmethod
    def is_inv_point(p):
        return np.isnan(p[0])

