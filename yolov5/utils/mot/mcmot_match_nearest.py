import numpy as np


def traj_sets_to_map(traj_sets, num_cams, inv_track_id):
    global_tracks = np.full((len(traj_sets), num_cams), inv_track_id, dtype=int)

    for global_track_idx, traj_set in enumerate(traj_sets):
        for cam_idx, local_track_id in traj_set:
            global_tracks[global_track_idx, cam_idx] = local_track_id

    return global_tracks


class MCTrajDisjSet:
    def __init__(self, local_track_ids, local_trajs, dist_thresh):
        num_cams = len(local_track_ids)
        # total_track_count = sum(cam_track_count)

        self.trajs_per_cam = np.array([ids.shape[0] for ids in local_track_ids])
        total_num_trajs = np.sum(self.trajs_per_cam)

        self.num_cams = num_cams
        self.cam_track_ranges = np.concatenate([[0], np.cumsum(self.trajs_per_cam)])

        # flatten
        self.local_traj_cams = np.concatenate([np.full(traj_count, cam_idx, dtype=np.int32) for cam_idx, traj_count in enumerate(self.trajs_per_cam)])
        self.local_track_ids = np.concatenate([ids for _, ids in enumerate(local_track_ids)])  # num_cam * traj_per_cam
        self.local_trajs = [traj for _, trajs in enumerate(local_trajs) for traj in trajs]

        self.dist_thresh = dist_thresh
        # self.parent_cams = [[(cam_idx == i) for i in range(num_cams)] for cam_idx, traj_count in enumerate(self.trajs_per_cam) for t in range(traj_count)]
        self.parent_cams = np.array([(0x1 << cam_idx) for cam_idx in self.local_traj_cams], dtype=np.uint32)
        assert num_cams <= 32

        self.rank = np.ones(total_num_trajs, dtype=np.int32)
        self.parent = np.arange(total_num_trajs, dtype=np.int32)
 
    def _has_same_cam(self, par_a, par_b):
        cams_a, cams_b = self.parent_cams[par_a], self.parent_cams[par_b]

        # TODO: use bitwise and for speed to check camera overlap
        # return any((has_cam_a and has_cam_b) for has_cam_a, has_cam_b in zip(cams_a, cams_b))
        return (cams_a & cams_b) != 0

    def _add_cams(self, par, child):
        cams_a, cams_b = self.parent_cams[par], self.parent_cams[child]

        # TODO: use bitwise or for speed
        # self.parent_cams[par] = [(has_cam_a or has_cam_b) for has_cam_a, has_cam_b in zip(cams_a, cams_b)]
        self.parent_cams[par] = cams_a | cams_b

    def _get_traj_cam(self, x):
        return self.local_traj_cams[x]

    def _get_traj(self, x):
        return self.local_trajs[x]

    def _get_traj_id(self, x):
        return self.local_track_ids[x]

    def _compute_traj_dist(self, x, y):
        traj_a = self._get_traj(x)
        traj_b = self._get_traj(y)

        dist = traj_a.distance_to(traj_b)
        return dist

    def find(self, x):
        if (self.parent[x] != x):
            self.parent[x] = self.find(self.parent[x])
 
        return self.parent[x]
 
    def union(self, x, y):
        if x == y or self._get_traj_cam(x) == self._get_traj_cam(y):
            # nodes are the same or have same camera
            # return False
            assert False

        xset = self.find(x)
        yset = self.find(y)
 
        if xset == yset or self._has_same_cam(xset, yset):
            return False
 
        traj_dist = self._compute_traj_dist(x, y)
        if traj_dist > self.dist_thresh:
            return False

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

        return True

    def union_all(self):
        for cam_a in range(self.num_cams):
            for cam_b in range(cam_a+1, self.num_cams):
                for i in range(self.cam_track_ranges[cam_a], self.cam_track_ranges[cam_a+1]):
                    for j in range(self.cam_track_ranges[cam_b], self.cam_track_ranges[cam_b+1]):
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
        return traj_sets_to_map(self.get_traj_sets(), num_cams, inv_track_id)


def mcmot_match_nearest(cams, matching_near_thresh, inv_track_id=-1):
    # TODO: to avoid ambiguities, might need to sort by spatial position or track id
    # visited = [[False] * cam.get_num_track() for cam in self.cams]

    local_track_ids = [np.array(list(cam.get_traj_dict().keys()), dtype=np.int32) for cam in cams]
    local_trajs = [list(cam.get_traj_dict().values()) for cam in cams]

    traj_disj_set = MCTrajDisjSet(local_track_ids, local_trajs, matching_near_thresh)
    traj_disj_set.union_all()
    global_tracks = traj_disj_set.get_traj_mapping(len(cams), inv_track_id)

    return global_tracks
