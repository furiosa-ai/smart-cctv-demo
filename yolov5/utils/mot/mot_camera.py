from typing import OrderedDict
import numpy as np

from calibration.util.camera_calib_node import CameraCalibrationNode
from utils.logging import log_func
from utils.mot.smoothing import SmoothingConv, SmoothingEMA, SmoothingNo, SmoothingSmoothDamp
from utils.mot.trajectory import Trajectory


class MOTCamera:
    inv_det = np.full(5, np.nan, dtype=float)

    def __init__(self, traj_timeout=np.inf, max_traj_count=None, max_traj_length=None, comp_trajs=True, 
        box_buffer_size=10, smooth_timeout=False, box_smoothing=False) -> None:  # , cam_calib: CameraCalibrationNode
        # self.cam_calib = cam_calib
        # self.boxes = []
        self.cur_frame_idx = -1
        self.time = 0

        self.trajs = {}  # TODO: refactor to 1D id array and 2d content array (for faster memory access)
        
        self.traj_timeout = traj_timeout
        # self.traj_timeout = 50
        self.comp_trajs = comp_trajs
        self.box_buffer_size = box_buffer_size
        self.max_traj_length = max_traj_length
        self.smooth_timeout = smooth_timeout  # will shift traj back and delete its starting point for each inv det that is added
        self.cur_boxes = None
        self.max_traj_count = max_traj_count

        # self.box_smoothing = SmoothingEMA(0.5)
        # self.box_smoothing = SmoothingConv(3)
        self.box_smoothing = SmoothingSmoothDamp(1) if box_smoothing else None

    def remove_traj(self, traj_id):
        del self.trajs[traj_id]

    def _smooth_boxes(self, boxes):
        if self.box_smoothing is not None:
            bb, track_ids = boxes[:, :4], boxes[:, 4]
            bb_smooth = self.box_smoothing(bb, track_ids)
            out = np.concatenate([bb_smooth, track_ids[:, None]], 1)
            return out
        else:
            return boxes

    @log_func
    def add_det(self, boxes, deproj_points, img_size=None):
        if img_size is not None:
            w, h = img_size
            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h

        # self.boxes.append(boxes)

        # if len(self.boxes) > self.box_buffer_size:
        #     self.boxes = self.boxes[-self.box_buffer_size:]

        if not self.comp_trajs or deproj_points is None:
            # self.trajs = {k: None for k in boxes[:, 4]}  # record ids only
            if self.max_traj_count is not None and len(boxes) > self.max_traj_count:
                track_indices = np.argsort(boxes[:, 4])
                track_indices = track_indices[:self.max_traj_count]
                boxes = boxes[track_indices]
        else:
            cur_track_ids = set([])
            timeout_traj = []
            new_trajs = []

            if len(boxes) > 0:
                # points = self._deproject_boxes(boxes)
                for track_id, point in zip(boxes[:, 4], deproj_points):
                    track_id = int(track_id)

                    assert track_id >= 0
                    if track_id not in self.trajs:
                        self._create_new_traj(track_id, point)
                        new_trajs.append(track_id)
                    else:
                        self.trajs[track_id].append(point)

                    cur_track_ids.add(track_id)

            # add a invalid det to trajectories not continued in this frame
            for track_id, traj in self.trajs.items():
                is_missing_traj = track_id not in cur_track_ids

                if is_missing_traj:
                    # timeout_traj.append(track_id)
                    # """
                    if not self.smooth_timeout:
                        traj.append_inv()
                    else:
                        traj.shift_back()

                if self.max_traj_length is not None:
                    traj.shorten(self.max_traj_length)

                if is_missing_traj:
                    if self.traj_timeout is not None:
                        if traj.is_timeout(self.traj_timeout):
                            timeout_traj.append(track_id)
                    # """

            # remove trajectories that are timeout
            for track_id in timeout_traj:
                del self.trajs[track_id]

            if self.max_traj_count is not None and self.num_traj > self.max_traj_count:
                for track_id in reversed(new_trajs):
                    if self.num_traj <= self.max_traj_count:
                        break
                    del self.trajs[track_id]

                valid_box_indices = [i for i in range(len(boxes)) if int(boxes[i, 4]) in self.trajs]
                boxes = boxes[valid_box_indices]

        self.cur_boxes = self._smooth_boxes(boxes)
        self.time += 1

    def _create_new_traj(self, track_id, start_point):
        self.trajs[track_id] = Trajectory([start_point], self.time)

    # @property
    # def cur_boxes(self):
    #     return self.boxes[-1]

    @property
    def num_frames(self):
        return self.time

    @property
    def num_traj(self):
        return len(self.trajs)

    """
    def get_box(self, frame_idx, track_id):
        boxes = self.boxes[frame_idx]
        return next((box for box in boxes if box[4] == track_id), None)
    """

    def get_box_track(self, track_id, rem_inv=False):
        track = [self.get_box(f, track_id) for f in range(self.num_frames)]

        if rem_inv:
            track = [b for b in track if b is not None]

        return track

    """
    def _deproject_boxes(self, boxes=None):
        if boxes is None:
            boxes = self.cur_boxes

        objs_x = (boxes[:, 0] + boxes[:, 2]) / 2
        objs_y = boxes[:, 3]

        points = np.stack([objs_x * self.cam_calib.calib_size[0], objs_y * self.cam_calib.calib_size[1]], 1)
        p3d, _ = self.cam_calib.deproject_to_ground(points, "z", omit_up_coord=True)

        return p3d
    """

    """
    def _deproject_traj(self, track_id):
        boxes = [next((box for box in frame_boxes if box[4] == track_id), MOTCamera.inv_det) for frame_boxes in self.boxes]
        boxes = np.stack(boxes)
        
        return self._deproject_boxes(boxes)
    """
    
    def get_track_ids(self, current_only=False):
        if current_only:
            return self.cur_boxes[:, 4].astype(int)
        else:
            return list(self.trajs.keys())

    def get_traj_dict(self):
        return self.trajs

    def get_traj(self, track_id, deprojected=True, points_only=False):
        # TODO: expensive
        assert deprojected
        # pos3d, track_ids = self.get_tracks()
        # return pos3d[np.where(track_ids == track_id)[0]]
        # return self._deproject_traj(track_id)

        if track_id not in self.trajs:
            return None

        traj = self.trajs[track_id]

        return traj if not points_only else traj.points

    def get_tracks(self, deprojected=True):
        assert deprojected

        pos3d = self.deproject_boxes()
        track_ids = self.get_track_ids()

        return pos3d, track_ids

    def get_num_track(self, current_only=False):
        # TODO: might have less track than boxes
        return len(self.get_track_ids(current_only))

    def print_traj_state(self, start_time=0, end_time=None):
        for track_id in self.get_track_ids():
            traj = self.get_traj(track_id).align(start_time, end_time)
            traj_str = "".join(["x" if Trajectory.is_inv_point(p) else "-" for p in traj])

            print(f"{track_id: 4d}: {traj_str}")
