# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.math cimport isnan

from libcpp.vector cimport vector

np.import_array()

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
# ctypedef enum DIST_NORM: DIST_NORM_L1, DIST_NORM_L2, DIST_NORM_L2SQ
# ctypedef enum DIST_REDUCE: DIST_REDUCE_MEDIAN, DIST_REDUCE_MEAN, DIST_REDUCE_LAST


ctypedef enum DIST_FUNC:
    DIST_FUNC_NP = 0
    DIST_FUNC_C = 1


ctypedef enum DIST_NORM:
    DIST_NORM_L1 = 0
    DIST_NORM_L2 = 1
    DIST_NORM_L2SQ = 2


ctypedef enum DIST_REDUCE:
    DIST_REDUCE_MEDIAN = 0
    DIST_REDUCE_MEAN = 1
    DIST_REDUCE_LAST = 2


cdef class TrajectoryC:
    cdef:
        DIST_NORM dist_norm
        DIST_REDUCE dist_reduce
        DIST_FUNC dist_func
        vector[float] points
        int start_time
        np.ndarray inv_point

        np.ndarray[DTYPE_t, ndim=2] get_points_c(self)
        float _dist_norm(self, np.ndarray[DTYPE_t, ndim=2] points1, np.ndarray[DTYPE_t, ndim=2] points2)
        int num_points(self)

        float distance_to_np(self, TrajectoryC traj)
        float distance_to_c(self, TrajectoryC traj)
        int get_start_time(self)

    cpdef float distance_to(self, TrajectoryC traj)
    cpdef DIST_NORM get_dist_norm_type(self)
    cpdef DIST_REDUCE get_dist_reduce_type(self)