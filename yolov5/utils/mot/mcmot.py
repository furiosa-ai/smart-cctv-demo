import time
import numpy as np

from calibration.util.axes_plot_3d import AxesPlot3d
from utils.logging import log_func
# from .mcmot_match_nearest import mcmot_match_nearest
from .mot_util import BiList
from .trajectory import Trajectory


class MCMOTBase:
    inv_track_id = -1

    def __init__(self, traj_timeout) -> None:
        self.traj_timeout = traj_timeout

    def update(self, cams):
        raise NotImplementedError()
    
    def get_global_track_ids(self):
        raise NotImplementedError()

    def get_traj(self, global_track_id, points_only=True):
        raise NotImplementedError()

    def local_to_global_track_id(self, cam_idx, local_track_id):
        raise NotImplementedError()

    def get_boxes(self, cam_idx, cam):
        boxes = cam.cur_boxes.copy()

        for bi in range(len(boxes)):
            ti = self.local_to_global_track_id(cam_idx, boxes[bi, 4])

            # if traj compuation is disabled for a cam just return the local as the global track id
            if ti != MCMOT.inv_track_id:
                boxes[bi, 4] = ti

        return boxes

    def __call__(self, cams):
        self.update(cams)

    @log_func
    def query_trajs(self, cams, track_type="local", track_ids=None, color_by="local_id", time_start=None, time_end=None, cam_indices=None, as_3d=False):
        from utils.colors import colors
        
        def _slice(idx, length=None):
            if idx is None:
                idx = None if length is None else (range(length) if isinstance(length, int) else length)
            elif not isinstance(idx, (tuple, list)):
                idx = [idx]
            return idx

        tracks3d = []

        if track_type == "local":
            cam_indices = _slice(cam_indices, len(cams))
            for cam_idx in cam_indices:
                cam = cams[cam_idx]
                cam_track_ids = _slice(track_ids, cam.get_track_ids())
                for track_id in cam_track_ids:
                    traj = cam.get_traj(track_id, points_only=False)
                    if traj is not None:
                        if time_start is not None:
                            traj = traj.align(time_start, time_end)
                        else:
                            traj = traj.points

                        tracks3d.append((cam_idx, track_id, traj))
        elif track_type == "global":
            if color_by == "global_id":
                color_by = "local_id"

            global_track_ids = _slice(track_ids, self.get_global_track_ids())
            for track_id in global_track_ids:
                traj = self.get_traj(track_id, points_only=False)
                if time_start is not None:
                    traj = traj.align(time_start, time_end)
                else:
                    traj = traj.points

                tracks3d.append((-1, track_id, traj))
        else:
            raise Exception(track_type)

        # filter zero length tracks
        tracks3d = [t for t in tracks3d if len(t[2]) > 0]

        trajs, color_ids = [], np.zeros(len(tracks3d), dtype=np.uint32)

        for i, (cam_idx, local_track_id, track3d) in enumerate(tracks3d):
            if color_by == "cam":
                color_id = cam_idx
            elif color_by == "local_id":
                color_id = local_track_id
            elif color_by == "global_id":
                global_track_id = self.local_to_global_track_id(cam_idx, local_track_id)
                color_id = global_track_id
            else:
                raise Exception(color_by)

            # color = np.array(color) / 255

            if as_3d:
                track3d = np.concatenate([track3d, np.zeros_like(track3d[:, :1])], 1)

            trajs.append(track3d)
            color_ids[i] = color_id

            # ax.plot(*track3d.T, color=color)

        return trajs, color_ids


class MCMOT(MCMOTBase):
    inv_track_id = MCMOTBase.inv_track_id

    def __init__(self, num_cams, matching_func="nearest", traj_timeout=1, matching_thresh=2*10, *args, **kwargs) -> None:
        super().__init__(traj_timeout=traj_timeout)
        
        self.num_cams = num_cams
        self.global_track_ids = None
        self.global_track_map = None  # maps each global track to its local tracks
        self.global_track_trajs = None

        self.matching_func = matching_func
        self.local_global_track_id_map = [{} for _ in range(self.num_cams)]  # cams x local_id -> global_id
        self.next_global_track_id = 0
        # self.matching_func = "same"

        # self.matching_near_thresh = 5 ** 2
    
        self.matching_near_thresh = matching_thresh ** 2  # WildTrack uses cm as unit

    # needs to be called every frame
    @log_func
    def update(self, cams):
        # TODO: only update current tracks (tracks that changed)
        t1 = time.time()
        global_track_map = self.match_local_tracks(cams, )
        t2 = time.time()
        global_track_ids = self.match_global_tracks(cams, global_track_map)
        t3 = time.time()

        global_track_trajs = self.compute_global_track_trajs(cams, global_track_map)
        t4 = time.time()

        assert len(global_track_map) == len(global_track_ids)
        assert len(global_track_map) == len(global_track_trajs)

        self.global_track_map = global_track_map
        self.global_track_ids = global_track_ids
        self.global_track_trajs = global_track_trajs

        if self.traj_timeout is not None:
            self.remove_timeout_trajs(cams)
        t5 = time.time()

        print("tracks: {}, match_local_tracks: {}, match_global_tracks: {}, compute_global_track_trajs: {}, rem_traj: {}".format(len(self.global_track_ids), *[f"{d*1e3}ms" for d in (
            (t2 - t1),
            (t3 - t2),
            (t4 - t3),
            (t5 - t4)
        )]))

    @log_func
    def remove_timeout_trajs(self, cams):
        timeout_traj_ids = []

        for traj_idx, traj_id in enumerate(self.global_track_ids):
            traj = self.global_track_trajs[traj_idx]
            if traj.is_timeout(self.traj_timeout):
                timeout_traj_ids.append(traj_id)

        self._remove_trajs(cams, timeout_traj_ids)

    def _remove_trajs(self, cams, global_track_ids):
        global_track_indices = [self.global_track_ids.get_index_of(id) for id in global_track_ids]
        global_track_indices = sorted(global_track_indices)

        # delete all respective local trajectories
        for global_track_idx in global_track_indices:
            for cam_idx, local_track_id in enumerate(self.global_track_map[global_track_idx]):
                if local_track_id != MCMOT.inv_track_id:
                    cams[cam_idx].remove_traj(local_track_id)

        self.global_track_map = np.delete(self.global_track_map, global_track_indices, 0)
        self.global_track_ids = self.global_track_ids.delete_range(global_track_indices)

        for index in reversed(global_track_indices):
            del self.global_track_trajs[index]

        assert len(self.global_track_map) == len(self.global_track_ids)
        assert len(self.global_track_map) == len(self.global_track_trajs)

    def _create_empty_track_map(self, size):
        return np.full((size, self.num_cams), MCMOT.inv_track_id, dtype=int)

    """
    # TODO: move to mcmot_traj_disj_set.py
    def _match_nearest(self):
        # TODO: to avoid ambiguities, might need to sort by spatial position or track id
        # visited = [[False] * cam.get_num_track() for cam in cams]

        traj_disj_set = MCTrajDisjSet(cams, self.matching_near_thresh)
        traj_disj_set.union_all()
        
        traj_sets = traj_disj_set.get_sets()

        global_tracks = self._create_empty_track_map(len(traj_sets))

        for global_track_idx, traj_set in enumerate(traj_sets):
            for cam_idx, local_track_id in traj_set:
                global_tracks[global_track_idx, cam_idx] = local_track_id

        return global_tracks
    """

    @log_func
    def match_local_tracks(self, cams):
        # TODO: improve putting every track seperate
        if self.matching_func == "sep":
            total_num_track = sum(cam.get_num_track() for cam in cams)
            global_tracks = self._create_empty_track_map(total_num_track)

            i = 0
            for cam_idx, cam in enumerate(cams):
                num_track = len(cam.get_tracks()[1])
                global_tracks[i:i+num_track, cam_idx] = cam.get_tracks()[1]
                i += num_track
        elif self.matching_func == "same":
            unique_local_track_ids = np.unique(np.concatenate([np.array(cam.get_track_ids()) for cam in cams]))
            global_tracks = self._create_empty_track_map(len(unique_local_track_ids))

            for cam_idx, cam in enumerate(cams):
                local_track_ids = set(cam.get_track_ids())

                for global_track_idx, global_track_id in enumerate(unique_local_track_ids):
                    global_tracks[global_track_idx][cam_idx] = global_track_id if global_track_id in local_track_ids else MCMOT.inv_track_id
        elif self.matching_func == "nearest":
            from .mcmot_match_nearest_c import mcmot_match_nearest_c as mcmot_match_nearest
            global_tracks = mcmot_match_nearest(cams, self.matching_near_thresh, MCMOT.inv_track_id)
        else:
            raise Exception(self.matching_func)

        return global_tracks

        # TODO: naive
        # self.global_track_ids = global_track_ids
        # self.global_track_map = global_tracks

    @log_func
    def compute_global_track_trajs(self, cams, global_track_map):
        global_track_trajs = []

        # compute global track trajectories (positions) from local tracks
        for global_track_idx, local_track_ids in enumerate(global_track_map):
            local_trajs = [cam.get_traj(local_track_id) for cam, local_track_id in zip(cams, local_track_ids) if local_track_id != MCMOT.inv_track_id]
            # mean_traj = np.nanmean(local_trajs, 0)
            mean_traj = Trajectory.mean(local_trajs)

            # check for timeouts
            # if the computed global trajectory is timeout (end contains nan x times -> all local traj contain nan x times)
            # delete global trajectory and all local trajectories
            global_track_trajs.append(mean_traj)

        return global_track_trajs

    @log_func
    def match_global_tracks(self, cams, cur_global_track_map):
        # match new to old tracks'
        # 1. keep old tracks even if not present in current tracks
        # 2. find matching ids
        # 3. assign new ids
        # assign new global id if no old line entry matches a new one

        new_global_track_ids = BiList()
        assigned_ids = {}  # global_id -> cam

        # TODO: make sure global_track_map is sorted by cam_idx otherwise id switches might occur
        # if sorted by cam_idx, camera 0 allways receives lower ids
        # if not sorted cameras might switch ids
        for cur_local_track_ids in cur_global_track_map:
            # search for cameras with matched local tracks
            global_id_candidates = ((cam_idx, self.local_to_global_track_id(cam_idx, local_track_id))
                for cam_idx, local_track_id in enumerate(cur_local_track_ids)
                if local_track_id != MCMOT.inv_track_id
            )
            
            # from first to last camera, search id that is found in last map and is not already assigned (or if assigned give priority to lower camera idx)
            cam_idx, global_id = next(((cam_idx, gid) for cam_idx, gid in global_id_candidates if gid != MCMOT.inv_track_id and (gid not in assigned_ids or cam_idx < assigned_ids[gid])), 
                (None, MCMOT.inv_track_id))

            # track id is not found (new track or splitted track)
            if global_id == MCMOT.inv_track_id:
                # first valid camera
                cam_idx = next(cam_idx
                    for cam_idx, local_track_id in enumerate(cur_local_track_ids)
                    if local_track_id != MCMOT.inv_track_id)
                global_id = self._new_global_track_id()
            
            new_global_track_ids.append(global_id)
            assigned_ids[global_id] = cam_idx

        return new_global_track_ids

    def _new_global_track_id(self):
        i = self.next_global_track_id
        self.next_global_track_id += 1
        return i

    def global_to_local_track_id(self, cam_idx, global_track_id):
        global_track_idx = self.global_track_ids.get_index_of(global_track_id)
        local_track_id = self.global_track_map[global_track_idx][cam_idx]
        return local_track_id

    def local_to_global_track_id(self, cam_idx, local_track_id):
        if self.global_track_map is None:
            return MCMOT.inv_track_id
        # return inv_track_id if not exist (might be new track)
        res = np.where(self.global_track_map[:, cam_idx] == local_track_id)[0]

        if len(res) == 0:
            return MCMOT.inv_track_id

        global_track_idx = res[0]
        global_track_id = self.global_track_ids[global_track_idx]
        return global_track_id

    def get_traj(self, global_track_id, points_only=True):
        global_track_idx = self.global_track_ids.get_index_of(global_track_id)
        traj = self.global_track_trajs[global_track_idx]

        return traj if not points_only else traj.points

    def get_global_track_ids(self):
        return self.global_track_ids


    """
    def draw_points(self, cams, frame_idx, camera_idx, track_id):
        def _slice(idx, length=None):
            if idx is None:
                idx = range(length) if length is not None else None
            elif not isinstance(idx, (tuple, list)):
                idx = [idx]
            return idx

        frame_idx = _slice(frame_idx, cams[0].num_frames)
        camera_idx = _slice(camera_idx, len(cams))
        track_id = _slice(track_id)

        cam_tracks = []

        for ci in camera_idx:
            tracks = []
            for ti in track_id:
                track = []
                for fi in frame_idx:
                    box = cams[ci].get_box(fi, ti)
                    if box is not None:
                        track.append(box)
                if len(track) > 0:
                    track = np.stack(track)
                tracks.append(track)  
            cam_tracks.append(tracks)

        if track_id is None:
            track_id = range(len(cams))
        elif not isinstance(track_id, (tuple, list)):
            track_id = [track_id]
    """
