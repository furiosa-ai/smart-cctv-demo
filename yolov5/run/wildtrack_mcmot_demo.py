import time
import numpy as np
import sys
import yaml
import glob
import cv2
import json
import os
import random
from calibration.util.camera_calib_node import CameraCalibrationNode
from utils.mot.mcmot_app_qt import MCMOTApp, MCMOTPipelineQT, run_mcmot_qt
from utils.mot.mcmot_pipeline import MCMOTPipeline
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout
from utils.mot.mcmot_stub import StubDetectorModelPool

from utils.mot.mot_pipeline import MOTPipeline, MOTPipelineOutput




class WildtrackDataset:
    def __init__(self, shuffle_box_order=True, id_filter=None, output_bgr=False) -> None:
        img_dirs = [
            "../data/Wildtrack_dataset/Image_subsets/C1",
            "../data/Wildtrack_dataset/Image_subsets/C2",
            "../data/Wildtrack_dataset/Image_subsets/C3",
            "../data/Wildtrack_dataset/Image_subsets/C4",
            "../data/Wildtrack_dataset/Image_subsets/C5",
            "../data/Wildtrack_dataset/Image_subsets/C6",
            "../data/Wildtrack_dataset/Image_subsets/C7",
        ]

        self.label_files = sorted(glob.glob("../data/Wildtrack_dataset/annotations_positions/*"))

        self.img_size = 1920, 1080
        self.num_cams = len(img_dirs)

        self.shuffle_box_order = shuffle_box_order
        self.id_filter = id_filter

        # last img does not have a label file
        self.img_dir_files = [sorted(glob.glob(os.path.join(d, "*")))[:-1] for d in img_dirs]
        self.num_imgs = len(self.img_dir_files[0])
        self.labels = [self._load_anno(i) for i in range(self.num_imgs)]
        self.output_bgr = output_bgr

        assert len(self) == len(self.labels)

    def _load_img(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (640, 360))
        if not self.output_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_anno(self, idx):
        path = self.label_files[idx]

        with open(path, "r") as f:
            annos = json.load(f)

        out_boxes = [[] for _ in range(self.num_cams)]

        for obj in annos:
            track_id = obj["personID"]

            for box in obj["views"]:
                xyxy = [box[k] for k in ("xmin", "ymin", "xmax", "ymax")]
                xyxy_track_id = xyxy + [track_id]

                if xyxy[0] != -1:
                    view_idx = box["viewNum"]

                    xyxy_track_id = np.float32(xyxy_track_id)
                    xyxy_track_id[[0, 2]] /= self.img_size[0]
                    xyxy_track_id[[1, 3]] /= self.img_size[1]

                    # boxes[view_idx] = xyxy
                    out_boxes[view_idx].append(xyxy_track_id)

            # out_track_ids.append(track_id)
            # out_boxes.append(boxes)

        # out_boxes = list(zip(*out_boxes))  # transpose

        if self.shuffle_box_order:
            for bi in range(len(out_boxes)):
                random.shuffle(out_boxes[bi])

        if self.id_filter is not None:
            for bi in range(len(out_boxes)):
                out_boxes[bi] = [box for box in out_boxes[bi] if int(box[4]) in self.id_filter]
                
        out_boxes = [(np.stack(b) if len(b) > 0 else np.zeros((0, 5))) for b in out_boxes]

        return out_boxes

    def get_image(self, cam_idx, idx):
        return self._load_img(self.img_dir_files[cam_idx][idx])

    def get_label(self, idx):
        return self._load_anno(idx)

    def __getitem__(self, idx):
        imgs = [self._load_img(f[idx]) for f in self.img_dir_files]
        boxes = self._load_anno(idx)

        return {
            "images": imgs,
            "boxes": boxes,  #  num_cam * num_obj
        }

    def __len__(self):
        return len(self.img_dir_files[0])


class MOTPipelineWildtrack:
    def __init__(self, cam_idx, calib, proj3d=True, *args, **kwargs):
        self.cam_idx = cam_idx
        self.dataset = WildtrackDataset(*args, **kwargs)
        self.cam_calib = calib
        self.frame_idx = 0
        self.proj3d = proj3d

    def _deproject_boxes(self, boxes):
        return MOTPipeline._deproject_boxes(self, boxes)

    def __call__(self):
        img = self.dataset.get_image(self.cam_idx, self.frame_idx)
        label = self.dataset.get_label(self.frame_idx)
        boxes_track = label[self.cam_idx]
        deproj_points = self._deproject_boxes(boxes_track)

        self.frame_idx += 1

        time.sleep(0.5)

        return MOTPipelineOutput(img, boxes_track, deproj_points)


class WildTrackMCMOTPipelineQT(MCMOTPipelineQT):
    def __init__(self, cfg=None, qt_signal=None) -> None:
        super().__init__(cfg=cfg, qt_signal=qt_signal)

    def _create_mot_pipeline(self, idx, model):
        mot_pipeline = MOTPipelineWildtrack(
            idx, 
            calib=CameraCalibrationNode.from_file(f"cfg/calib/wildtrack/cam_calib{idx}.yaml"),
            # id_filter=[9, 7, 40, 5, 49], 
            # id_filter=[99999999], 
            output_bgr=True)
        return mot_pipeline

    def _create_detector_model_pool(self):
        return StubDetectorModelPool()


def main():
    with open("cfg/wildtrack_mot_pipeline.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    run_mcmot_qt(cfg=cfg, pipeline_cls=WildTrackMCMOTPipelineQT)

    print("done")


if __name__ == "__main__":
    main()
