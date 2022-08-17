import numpy as np

from utils.mot.mcmot import MCMOT, MCMOTBase


class StubDetector:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return np.zeros((0, 6), dtype=np.float32)


class StubDetectorModelPool:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_inference(*args, **kwargs):
        return None

    def start(self):
        pass

    def exit(self):
        pass


class StubMCMOT(MCMOTBase):
    def __init__(self, traj_timeout=None, *args, **kwargs) -> None:
        super().__init__(traj_timeout)

        self.cams = None
        self.global_to_local_id_map = None
        self.local_to_global_map = None

        # TODO: need to implement traj del management, not handled by MOT since boxes from tracker are just appended to traj
        assert traj_timeout is None, "Traj management is handled by mot themselves"

    def update(self, cams):
        self.cams = cams

        self.global_to_local_id_map = {}
        self.local_to_global_map = {}

        i = 0
        for cam_idx, cam in enumerate(cams):
            for track_id in cam.get_track_ids():
                self.global_to_local_id_map[i] = (cam_idx, track_id)
                self.local_to_global_map[(cam_idx, track_id)] = i
                i += 1
    
    def get_global_track_ids(self):
        return list(self.global_to_local_id_map.keys())

    def get_traj(self, global_track_id, points_only=True):
        cam_idx, local_track_id = self.global_to_local_id_map[global_track_id]
        traj = self.cams[cam_idx].get_traj(local_track_id)

        return traj if not points_only else traj.points

    def local_to_global_track_id(self, cam_idx, local_track_id):
        return local_track_id
        # return self.local_to_global_map.get((cam_idx, local_track_id), local_track_id)
