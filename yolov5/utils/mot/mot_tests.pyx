# cython: binding=True

import numpy as np
cimport numpy as np
import pytest

from .trajectory import Trajectory, DistNormType, DistReduceType
from .trajectory_c cimport TrajectoryC, DIST_NORM_L1, DIST_NORM_L2, DIST_NORM_L2SQ, DIST_REDUCE_LAST, DIST_REDUCE_MEAN, DIST_REDUCE_MEDIAN, DIST_FUNC_NP, DIST_FUNC_C

from utils.cytest import cytest, assert_equal


np.import_array()

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


@cytest
def test_traj_distance_to1():
    points = np.zeros((3, 2))
    start_time = 0

    t1 = TrajectoryC.from_trajectory(Trajectory(points, start_time), dist_func=DIST_FUNC_NP)
    t2 = TrajectoryC.from_trajectory(Trajectory(points, start_time), dist_func=DIST_FUNC_NP)

    assert_equal(t1.distance_to(t2), 0)


@cytest
def test_traj_distance_to2():
    points = np.zeros((3, 2))
    start_time = 0

    t1 = TrajectoryC.from_trajectory(Trajectory(points, start_time), dist_func=DIST_FUNC_C)
    t2 = TrajectoryC.from_trajectory(Trajectory(points, start_time), dist_func=DIST_FUNC_C)

    assert_equal(t1.distance_to(t2), 0)


@cytest
def test_traj_distance_to3():
    np.random.seed(123)
    points1 = np.random.rand(3, 2)
    points2 = np.random.rand(3, 2)
    start_time1 = 0
    start_time2 = 0

    dist_norm = DistNormType.DIST_NORM_L2SQ
    dist_reduce = DistReduceType.DIST_REDUCE_MEDIAN

    t1_np = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_NP)
    t2_np = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_NP)

    t1_c = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)
    t2_c = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)

    assert_equal(t1_np.distance_to(t2_np), t1_c.distance_to(t2_c))


@cytest
def test_traj_distance_to4():
    np.random.seed(12)
    points1 = np.random.rand(7, 2)
    points2 = np.random.rand(7, 2)
    start_time1 = 0
    start_time2 = 0

    dist_norm = DistNormType.DIST_NORM_L2SQ
    dist_reduce = DistReduceType.DIST_REDUCE_MEAN

    t1_np = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_NP)
    t2_np = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_NP)

    t1_c = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)
    t2_c = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)

    assert_equal(t1_np.distance_to(t2_np), t1_c.distance_to(t2_c))



@cytest
def test_traj_distance_to5():
    np.random.seed(12)
    points1 = np.random.rand(7, 2)
    points2 = np.random.rand(7, 2)
    start_time1 = 0
    start_time2 = 0

    dist_norm = DistNormType.DIST_NORM_L2SQ
    dist_reduce1 = DistReduceType.DIST_REDUCE_MEDIAN
    dist_reduce2 = DistReduceType.DIST_REDUCE_MEAN

    t1 = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce1), dist_func=DIST_FUNC_NP)
    t2 = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce2), dist_func=DIST_FUNC_NP)

    assert_equal(t1.get_dist_reduce_type(), DIST_REDUCE_MEDIAN)
    assert_equal(t2.get_dist_reduce_type(), DIST_REDUCE_MEAN)


@cytest
def test_traj_distance_to6():
    np.random.seed(12)
    points1 = np.random.rand(7, 2)
    points2 = np.random.rand(7, 2)
    start_time1 = 0
    start_time2 = 0

    dist_norm = DistNormType.DIST_NORM_L2SQ
    dist_reduce = DistReduceType.DIST_REDUCE_MEDIAN

    t1_np = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_NP)
    t2_np = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_NP)

    t1_c = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)
    t2_c = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)

    assert_equal(t1_np.distance_to(t2_np), t1_c.distance_to(t2_c))


@cytest
def test_traj_distance_to7():
    np.random.seed(12)
    start_time1 = np.random.randint(0, 100)
    start_time2 = np.random.randint(0, 100)
    points1 = np.random.rand(100 - start_time1, 2)
    points2 = np.random.rand(100 - start_time2, 2)

    dist_norm = DistNormType.DIST_NORM_L2SQ
    dist_reduce = DistReduceType.DIST_REDUCE_MEDIAN

    t1_np = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_NP)
    t2_np = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_NP)

    t1_c = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)
    t2_c = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)

    assert_equal(t1_np.distance_to(t2_np), t1_c.distance_to(t2_c))


@cytest
def test_traj_distance_to8():
    np.random.seed(12)
    start_time1 = 0
    start_time2 = 0
    points1 = np.full((1, 2), np.nan)
    points2 = np.full((1, 2), 0)

    dist_norm = DistNormType.DIST_NORM_L2SQ
    dist_reduce = DistReduceType.DIST_REDUCE_MEDIAN

    t1_c = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)
    t2_c = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)

    assert_equal(t1_c.distance_to(t2_c), np.inf)


# if n is even, median returns avg of middle two elements
@cytest
def test_traj_distance_to9():
    np.random.seed(1234)
    start_time1 = np.random.randint(0, 100)
    start_time2 = np.random.randint(0, 100)
    points1 = np.random.rand(100 - start_time1, 2)
    points2 = np.random.rand(100 - start_time2, 2)

    points1[np.random.randint(100 - start_time1):] = np.nan
    points2[np.random.randint(100 - start_time2):] = np.nan

    dist_norm = DistNormType.DIST_NORM_L2SQ
    dist_reduce = DistReduceType.DIST_REDUCE_MEAN

    t1_np = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_NP)
    t2_np = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_NP)

    t1_c = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)
    t2_c = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)

    assert_equal(t1_np.distance_to(t2_np), t1_c.distance_to(t2_c))


@cytest
def test_traj_distance_to10():
    start_time1 = 0
    start_time2 = 0

    points1 = np.array([
        [0, 0],
        [0, 0],])
    points2 = np.array([
        [1, 1],
        [2, 2],])

    dist_norm = DistNormType.DIST_NORM_L2SQ
    dist_reduce = DistReduceType.DIST_REDUCE_MEAN

    t1_np = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_NP)
    t2_np = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_NP)

    t1_c = TrajectoryC.from_trajectory(Trajectory(points1, start_time1, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)
    t2_c = TrajectoryC.from_trajectory(Trajectory(points2, start_time2, dist_norm, dist_reduce), dist_func=DIST_FUNC_C)

    assert_equal(t1_np.distance_to(t2_np), t1_c.distance_to(t2_c))