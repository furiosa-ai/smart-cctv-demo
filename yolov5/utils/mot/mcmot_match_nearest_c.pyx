# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False
# distutils: language = c++

import time
import numpy as np
cimport numpy as np

from .trajectory_c cimport TrajectoryC


np.import_array()

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def traj_sets_to_map(traj_sets, num_cams, inv_track_id):
    global_tracks = np.full((len(traj_sets), num_cams), inv_track_id, dtype=int)

    for global_track_idx, traj_set in enumerate(traj_sets):
        for cam_idx, local_track_id in traj_set:
            global_tracks[global_track_idx, cam_idx] = local_track_id

    return global_tracks


cdef class MCTrajDisjSet:
    cdef:
        float dist_thresh
        int num_cams
        int[:] trajs_per_cam
        unsigned int[:] cam_track_ranges
        int[:] local_traj_cams
        int[:] local_track_ids
        TrajectoryC[:] local_trajs
        unsigned int[:] parent_cams
        int[:] rank
        int[:] parent

    def __init__(self, local_track_ids, local_trajs, dist_thresh):
        num_cams = len(local_track_ids)
        # total_track_count = sum(cam_track_count)

        trajs_per_cam = np.array([ids.shape[0] for ids in local_track_ids], dtype=np.int32)
        total_num_trajs = np.sum(trajs_per_cam)

        self.dist_thresh = dist_thresh
        self.num_cams = num_cams
        self.trajs_per_cam = trajs_per_cam
        self.cam_track_ranges = np.concatenate([[0], np.cumsum(trajs_per_cam)]).astype(np.uint32)

        # flatten
        self.local_traj_cams = np.concatenate([np.full(traj_count, cam_idx, dtype=np.int32) for cam_idx, traj_count in enumerate(trajs_per_cam)])
        self.local_track_ids = np.concatenate([ids for ids in local_track_ids])  # num_cam * traj_per_cam
        self.local_trajs = np.array([TrajectoryC.from_trajectory(traj) for trajs in local_trajs for traj in trajs], dtype=TrajectoryC)

        self.parent_cams = np.array([(0x1 << cam_idx) for cam_idx in self.local_traj_cams], dtype=np.uint32)
        assert num_cams <= 32

        self.rank = np.ones(total_num_trajs, dtype=np.int32)
        self.parent = np.arange(total_num_trajs, dtype=np.int32)
 
    cdef bint _has_same_cam(self, int par_a, int par_b):
        cdef int cams_a = self.parent_cams[par_a]
        cdef int cams_b = self.parent_cams[par_b]

        # TODO: use bitwise and for speed to check camera overlap
        # return any((has_cam_a and has_cam_b) for has_cam_a, has_cam_b in zip(cams_a, cams_b))
        return (cams_a & cams_b) != 0

    cdef void _add_cams(self, int par, int child):
        cdef int cams_a = self.parent_cams[par]
        cdef int cams_b = self.parent_cams[child]

        # TODO: use bitwise or for speed
        # self.parent_cams[par] = [(has_cam_a or has_cam_b) for has_cam_a, has_cam_b in zip(cams_a, cams_b)]
        self.parent_cams[par] = cams_a | cams_b

    cdef int _get_traj_cam(self, int x):
        return self.local_traj_cams[x]

    cdef TrajectoryC _get_traj(self, int x):
        return self.local_trajs[x]

    cdef int _get_traj_id(self, int x):
        return self.local_track_ids[x]

    cdef float _compute_traj_dist(self, int x, int y):
        cdef TrajectoryC traj_a = self._get_traj(x)
        cdef TrajectoryC traj_b = self._get_traj(y)

        # cdef float dist = traj_a.distance_to(traj_b)
        cdef float dist = traj_a.distance_to(traj_b)
        return dist

    cdef int find(self, int x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
 
        return self.parent[x]
 
    cdef void union(self, int x, int y):
        # if x == y or self._get_traj_cam(x) == self._get_traj_cam(y):
            # nodes are the same or have same camera
            # return False
            # assert False

        cdef int xset = self.find(x)
        cdef int yset = self.find(y)
 
        if xset == yset or self._has_same_cam(xset, yset):
            return
 
        cdef float traj_dist = self._compute_traj_dist(x, y)
        if traj_dist > self.dist_thresh:
            return

        if self.rank[xset] < self.rank[yset]:
            self.parent[xset] = yset
            self._add_cams(yset, xset)
        elif self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset
            self._add_cams(xset, yset)
        else:
            self.parent[yset] = xset
            self._add_cams(xset, yset)
            self.rank[xset] = self.rank[xset] + 1

    cdef union_all(self):
        cdef:
            unsigned int num_cams, cam_a_start, cam_a_end, cam_b_start, cam_b_end, cam_a, cam_b, i, j
        
        num_cams = self.num_cams

        for cam_a in range(num_cams):
            cam_a_start = self.cam_track_ranges[cam_a]
            cam_a_end = self.cam_track_ranges[cam_a+1]

            for cam_b in range(cam_a+1, num_cams):
                cam_b_start = self.cam_track_ranges[cam_b]
                cam_b_end = self.cam_track_ranges[cam_b+1]

                for i in range(cam_a_start, cam_a_end):
                    for j in range(cam_b_start, cam_b_end):
                        self.union(i, j)

    def get_traj_sets(self):
        sets = {}

        for i in range(len(self.parent)):
            p = self.find(i)
            val = self._get_traj_cam(i), self._get_traj_id(i)

            if p not in sets:
                sets[p] = [val]
            else:
                sets[p].append(val)

        return list(sets.values())

    def get_traj_mapping(self, num_cams, inv_track_id):
        # t1 = time.time()
        mapping = traj_sets_to_map(self.get_traj_sets(), num_cams, inv_track_id)
        # print((time.time() - t1) * 1000)
        return mapping


def mcmot_match_nearest_c(cams, matching_near_thresh, inv_track_id=-1):
    # TODO: to avoid ambiguities, might need to sort by spatial position or track id
    # visited = [[False] * cam.get_num_track() for cam in self.cams]

    local_track_ids = [np.array(list(cam.get_traj_dict().keys()), dtype=np.int32) for cam in cams]
    local_trajs = [list(cam.get_traj_dict().values()) for cam in cams]

    traj_disj_set = MCTrajDisjSet(local_track_ids, local_trajs, matching_near_thresh)
    # t1 = time.time()
    traj_disj_set.union_all()
    # print((time.time() - t1) * 1000)
    global_tracks = traj_disj_set.get_traj_mapping(len(cams), inv_track_id)

    return global_tracks
