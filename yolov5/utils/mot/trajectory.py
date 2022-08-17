from enum import IntEnum
import numpy as np

# ctypedef enum DIST_NORM: DIST_NORM_L1, DIST_NORM_L2, DIST_NORM_L2SQ
# ctypedef enum DIST_REDUCE: DIST_REDUCE_MEDIAN, DIST_REDUCE_MEAN, DIST_REDUCE_LAST


class DistNormType(IntEnum):
    DIST_NORM_L1 = 0
    DIST_NORM_L2 = 1
    DIST_NORM_L2SQ = 2


class DistReduceType(IntEnum):
    DIST_REDUCE_MEDIAN = 0
    DIST_REDUCE_MEAN = 1
    DIST_REDUCE_LAST = 2


class Trajectory:
    # inv_point = np.full(2, np.nan, dtype=float)

    # dist_norm = DistNormType.DIST_NORM_L2SQ
    # dist_reduce = DistReduceType.DIST_REDUCE_MEDIAN

    def __init__(self, points, cur_time, dist_norm=DistNormType.DIST_NORM_L2SQ, dist_reduce=DistReduceType.DIST_REDUCE_MEDIAN) -> None:
        self.points = np.array(points)
        self.start_time = cur_time
        self.dist_norm = dist_norm
        self.dist_reduce = dist_reduce
        self.inv_point = np.full(self.points.shape[-1], np.nan, dtype=float)

    def __getitem__(self, i):
        return self.points[i]

    """
    def __setitem__(self, i, val):
        self.points[i] = val
    """

    def append(self, x):
        # self.points.append(x)
        self.points = np.append(self.points, x[None], 0)

    def append_inv(self):
        self.points = np.append(self.points, self.inv_point[None], 0)

    def shift_back(self):
        self.points = np.roll(self.points, -1, axis=0)
        self.points[-1] = self.inv_point
        # self.points = self.points[1:]
        # self.append_inv()
        self.start_time += 1

    def shorten(self, l):
        cur_l = self.points.shape[0]
        new_l = min(cur_l, l)
        self.points = self.points[-new_l:]
        self.start_time = self.start_time + (cur_l - new_l)

    def is_timeout(self, timeout):
        l = min(timeout, self.points.shape[0])
        return np.isnan(self.points[-l:, 0]).all()

    def _dist_norm(self, point, points):
        # n x l * 2

        if self.dist_reduce == DistReduceType.DIST_REDUCE_LAST:
            point = point[-1:]
            points = [traj[-1:] for traj in points]

        diffs = points - point[None]

        # TODO: if not overlapping, all entries will be nan in a subset of the array -> set distance to inf
        # otherwise ignore nan values

        if self.dist_norm == DistNormType.DIST_NORM_L2SQ:
            x = np.sum(diffs * diffs, 2)
        elif self.dist_norm == DistNormType.DIST_NORM_L2:
            x = np.sqrt(np.sum(diffs * diffs, 2))
        elif self.dist_norm == DistNormType.DIST_NORM_L1:
            x = np.sum(np.abs(diffs), 2)

        if self.dist_reduce == DistReduceType.DIST_REDUCE_MEDIAN:
            x = np.nanmedian(x, 1)
        elif self.dist_reduce == DistReduceType.DIST_REDUCE_MEAN:
            x = np.nanmean(x, 1)
        elif self.dist_reduce == DistReduceType.DIST_REDUCE_LAST:
            assert x.shape[1] == 1
            x = x[:, -1]

        x[np.isnan(x)] = np.inf

        return x

    def distance_to(self, traj):
        t = max(self.start_time, traj.start_time)

        p1 = self.points[t - self.start_time:]
        p2 = traj.points[t - traj.start_time:]

        assert len(p1) == len(p2)

        return self._dist_norm(p1, [p2])[0]

    def distances_to(self, trajs):
        t = max(self.start_time, max(traj.start_time for traj in trajs))

        p1 = self.points[t - self.start_time:]
        p2 = [traj.points[t - self.start_time:] for traj in trajs]

        assert all(len(p1) == len(p) for p in p2)

        return self.dist_norm(p1, p2)

    def align(self, start, end=None, points_only=True):
        assert points_only

        if start >= self.start_time:
            # cut
            p = self.points[start - self.start_time:]
        else:
            # pad
            p = np.concatenate([
                np.repeat(self.inv_point[None], self.start_time - start, 0),
                self.points
            ], 0)

        if end is not None:
            assert len(p) >= end - start
            p = p[:end - start]

        return p

    def __len__(self):
        return len(self.points)

    @staticmethod
    def mean(trajs):
        t = min(traj.start_time for traj in trajs)

        points = [traj.align(t) for traj in trajs]
        mean_traj = np.nanmean(points, 0)

        return Trajectory(mean_traj, t)

    @staticmethod
    def is_inv_point(p):
        return np.isnan(p[0])
