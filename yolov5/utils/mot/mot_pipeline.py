import numpy as np
import cv2
from collections import namedtuple
from calibration.util.camera_calib_node import CameraCalibrationNode
from utils.logging import log_func

from utils.mot.mot_camera import MOTCamera
from utils.mot.smoothing import SmoothingConv, SmoothingSmoothDamp
from utils.util import PerfMeasure


MOTPipelineOutput = namedtuple("MOTPipelineOutput", ("image", "boxes_track", "deproj_points"))


class MOTPipeline:
    def __init__(self, video_input, detector, tracker, calib, proj3d=False, 
        box_smoothing="default", traj_smoothing="default"):
        
        if box_smoothing == "default":
            box_smoothing = "SmoothingSmoothDamp(1)"
        
        if traj_smoothing == "default":
            traj_smoothing = "SmoothingConv(3)"

        self.video_input = video_input
        self.detector = detector
        self.tracker = tracker
        self.cam_calib = calib
        self.proj3d = proj3d
        self.box_smoothing = eval(box_smoothing) if box_smoothing is not None else None
        self.traj_smoothing = eval(traj_smoothing) if traj_smoothing is not None else None

        if self.video_input.size != self.cam_calib.calib_size:
            print(f"Scaling calib to {self.video_input.size} '{self.video_input.src}'")
            self.cam_calib.scale_to(self.video_input.size)

    @log_func
    def _smooth_boxes(self, boxes):
        if self.box_smoothing is not None:
            bb, track_ids = boxes[:, :4], boxes[:, 4]
            bb_smooth = self.box_smoothing(bb, track_ids)
            out = np.concatenate([bb_smooth, track_ids[:, None]], 1)
            return out
        else:
            return boxes

    @log_func
    def _smooth_trajs(self, points, ids):
        if self.traj_smoothing is not None:
            points = self.traj_smoothing(points, ids)
        
        return points

    @staticmethod
    def get_output_format(cam_cfg):
        w, h = cam_cfg["input"]["size"]

        fmt = MOTPipelineOutput(
            image=np.zeros((h, w, 3), dtype=np.uint8),
            boxes_track=None,
            deproj_points=None
        )

        shm_supported = [True, False, False]

        return fmt, shm_supported

    @log_func
    def _read_img(self):
        return self.video_input()

    @log_func
    def _deproject_boxes(self, boxes):
        if len(boxes) > 0:
            sx, sy = self.cam_calib.calib_size

            objs_x = (boxes[:, 0] + boxes[:, 2]) / 2
            objs_y = boxes[:, 3]
    
            points = np.stack([objs_x * sx, objs_y * sy], 1)

            if self.proj3d:
                objs_heighs = (boxes[:, 3] - boxes[:, 1]) * sy
                p3d = self.cam_calib.deproject_vert_lines_to_ground(points, objs_heighs, "z")
            else:
                p3d, _ = self.cam_calib.deproject_to_ground(points, "z", omit_up_coord=True)
        else:
            p3d = []

        return p3d

    @log_func
    def _undistort(self, img):
        return self.cam_calib.process_image(img)

    @log_func
    def __call__(self):
        img = self._read_img()

        # undistort image in case
        if img is None:
            boxes_track = []
            deproj_points = []
        else:
            img = self._undistort(img)
            # img = cv2.flip(img, 1)

            boxes_det = self.detector(img)

            with PerfMeasure("Tracking"):
                boxes_track = self.tracker(img, boxes_det)

            if len(boxes_track) == 0:
                boxes_track = np.zeros((0, 6), dtype=np.float32)

            # normalize to 0-1
            boxes_track[:, [0, 2]] /= img.shape[1]
            boxes_track[:, [1, 3]] /= img.shape[0]

            boxes_track = self._smooth_boxes(boxes_track)
            deproj_points = self._deproject_boxes(boxes_track)
            deproj_points = self._smooth_trajs(deproj_points, boxes_track[:, 4])

        return MOTPipelineOutput(img, boxes_track, deproj_points)
