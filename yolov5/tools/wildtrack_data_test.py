

from collections import namedtuple
import glob
import random
import sys
import os
import time
from xml.dom import minidom
import xml.etree.ElementTree as ElementTree
import cv2
import json
import numpy as np
from utils.colors import colors

# from utils.mot import MCMOT, MOTCamera
from utils.mot import MCMOT, MOTCamera

sys.path.append("calibration")
from calibration.util.axes_plot_3d import AxesPlot3d
from calibration.util.camera_calib_node import CameraCalibrationNode
from calibration.util.util import IntrinsicCalibrationData


class WildtrackDataset:
    def __init__(self, shuffle_box_order=True, id_filter=None) -> None:
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

        assert len(self) == len(self.labels)

    def _load_img(self, path):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 360))
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


def draw_camera(img, boxes):
    ih, iw = img.shape[:2]
    for *box, track_id in boxes:
        # if box is not None:
        box = np.array(box)

        box[0::2] *= iw
        box[1::2] *= ih

        x1, y1, x2, y2 = box.astype(int)
        color = colors(track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, "{}".format(int(track_id)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return img


def vis_data_sample(sample):
    imgs = [img.copy() for img in sample["images"]]

    for img, boxes in zip(imgs, sample["boxes"]):
        ih, iw = img.shape[:2]
        for *box, track_id in boxes:
            # if box is not None:
            box = np.array(box)

            box[0::2] *= iw
            box[1::2] *= ih

            x1, y1, x2, y2 = box.astype(int)
            color = colors(track_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, "{}".format(int(track_id)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    imgs.append(np.zeros_like(imgs[0]))

    img_grid = np.concatenate([
        np.concatenate(imgs[:4], 1),
        np.concatenate(imgs[4:], 1)
    ], 0)

    img_grid = cv2.cvtColor(img_grid, cv2.COLOR_RGB2BGR)
    cv2.imshow("out", img_grid)
    cv2.waitKey(0)


def load_opencv_xml(filename, element_name, dtype='float32'):
    """
    Loads particular element from a given OpenCV XML file.

    Raises:
        FileNotFoundError: the given file cannot be read/found
        UnicodeDecodeError: if error occurs while decoding the file

    :param filename: [str] name of the OpenCV XML file
    :param element_name: [str] element in the file
    :param dtype: [str] type of element, default: 'float32'
    :return: [numpy.ndarray] the value of the element_name
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError("File %s not found." % filename)

    tree = ElementTree.parse(filename)
    rows = int(tree.find(element_name).find('rows').text)
    cols = int(tree.find(element_name).find('cols').text)
    return np.fromstring(tree.find(element_name).find('data').text,
                            dtype, count=rows*cols, sep=' ').reshape((rows, cols))


def load_camera_node(intr_calib_file, extr_calib_file):
    rvec, tvec = [], []

    size = 1920, 1080
    K = load_opencv_xml(intr_calib_file, 'camera_matrix')
    D = load_opencv_xml(intr_calib_file, 'distortion_coefficients')

    intr = IntrinsicCalibrationData(K, D, size)

    xmldoc = minidom.parse(extr_calib_file)
    rvec.append([float(number)
                    for number in xmldoc.getElementsByTagName('rvec')[0].childNodes[0].nodeValue.strip().split()])
    tvec.append([float(number)
                    for number in xmldoc.getElementsByTagName('tvec')[0].childNodes[0].nodeValue.strip().split()])

    cam_calib = CameraCalibrationNode(intr)
    cam_calib.set_pose_from_rvec_tvec(rvec, tvec)

    return MOTCamera(cam_calib)



def _build_mcmot(id_filter=None, *args, **kwargs):
    intr_calib_files = [
        "../data/Wildtrack_dataset/calibrations/intrinsic_zero/intr_CVLab1.xml",
        "../data/Wildtrack_dataset/calibrations/intrinsic_zero/intr_CVLab2.xml",
        "../data/Wildtrack_dataset/calibrations/intrinsic_zero/intr_CVLab3.xml",
        "../data/Wildtrack_dataset/calibrations/intrinsic_zero/intr_CVLab4.xml",
        "../data/Wildtrack_dataset/calibrations/intrinsic_zero/intr_IDIAP1.xml",
        "../data/Wildtrack_dataset/calibrations/intrinsic_zero/intr_IDIAP2.xml",
        "../data/Wildtrack_dataset/calibrations/intrinsic_zero/intr_IDIAP3.xml",
    ]

    extr_calib_files = [
        "../data/Wildtrack_dataset/calibrations/extrinsic/extr_CVLab1.xml",
        "../data/Wildtrack_dataset/calibrations/extrinsic/extr_CVLab2.xml",
        "../data/Wildtrack_dataset/calibrations/extrinsic/extr_CVLab3.xml",
        "../data/Wildtrack_dataset/calibrations/extrinsic/extr_CVLab4.xml",
        "../data/Wildtrack_dataset/calibrations/extrinsic/extr_IDIAP1.xml",
        "../data/Wildtrack_dataset/calibrations/extrinsic/extr_IDIAP2.xml",
        "../data/Wildtrack_dataset/calibrations/extrinsic/extr_IDIAP3.xml",
    ]

    camera_nodes = [load_camera_node(i, e) for i, e in zip(intr_calib_files, extr_calib_files)]

    for i, cam_node in enumerate(camera_nodes):
        cam_node.cam_calib.save(f"cfg/calib/wildtrack/cam_calib{i}.yaml")

    data = WildtrackDataset(id_filter=id_filter)

    # sample = data[0]
    # vis_data_sample(sample)

    # 

    mcmot = MCMOT(len(camera_nodes), *args, **kwargs)

    return  data, mcmot, camera_nodes


def _test_show_images():
    data, mcmot = _build_mcmot()

    for i in range(len(data)):
        sample = data[i]
        vis_data_sample(sample)


def _test_sim_traj():
    data, mcmot = _build_mcmot(matching_func="same")

    # end_time = len(data)
    end_time = 120
    start_end = 0
    for frame_idx in range(end_time):
        label = data.get_label(frame_idx)
        for cam_idx, cam in enumerate(mcmot.cams):
            cam.add_det(label[cam_idx])

        t1 = time.time()
        mcmot.update()
        t2 = time.time()

    ax = AxesPlot3d()

    # cam_idx = 0

    for t in range(start_end, end_time, 1):
        ax.clear()
        # mcmot.draw_traj(ax, track_type="local", color_by="global_id", time_end=t, cam_indices=[cam_idx])
        # mcmot.draw_traj(ax, track_type="local", color_by="global_id", time_end=t)
        # mcmot.draw_traj(ax, track_type="local", color_by="cam", time_end=t)
        mcmot.draw_traj(ax, track_type="global", color_by="global_id", time_end=t) 
        ax.show(pause=0.02)
        ax.save(f"out/plt/{t:04d}.jpg")
        # ax.save(f"out/plt/{cam_idx}.jpg")

    ax.show(True)


def _test_local_vs_global_id():
    data, mcmot = _build_mcmot()

    for frame_idx in range(data.num_imgs):
        label = data.get_label(frame_idx)
        for cam_idx, cam in enumerate(mcmot.cams):
            cam.add_det(label[cam_idx])

        t1 = time.time()
        mcmot.update()
        t2 = time.time()

        img = data.get_image(0, frame_idx)

        img_local = img.copy()
        img_global = img.copy()

        boxes = label[0]
        draw_camera(img_local, boxes)
        boxes = mcmot.get_boxes(0)
        draw_camera(img_global, boxes)

        img = np.concatenate([img_local, img_global], 1)

        cv2.imshow("out", img)
        cv2.waitKey(0)

        # print(f"Matching took {(t2 - t1) * 1e3}ms")


def _test_few_target_track():
    # id_filter = [9]
    # id_filter = None
    id_filter = [9, 7, 40, 5, 49]

    data, mcmot, cams = _build_mcmot(id_filter=id_filter)

    ax_local = AxesPlot3d(num_col=2)
    ax_global = AxesPlot3d(num_col=2)

    for frame_idx in range(data.num_imgs):
        label = data.get_label(frame_idx)
        for cam_idx, cam in enumerate(cams):
            cam.add_det(label[cam_idx])

        t1 = time.time()
        boxes = mcmot.update(cams)
        t2 = time.time()

        # global_track_id = mcmot.local_to_global_track_id(local_cam_idx0, local_track_id0)

        img = data.get_image(0, frame_idx)

        img_local = img.copy()
        img_global = img.copy()

        boxes = label[0]
        draw_camera(img_local, boxes)
        boxes = mcmot.get_boxes(0, cams[0])
        draw_camera(img_global, boxes)

        img = np.concatenate([img_local, img_global], 1)

        ax_local.clear()
        # ax_global.clear()
        mcmot.draw_traj(cams, ax_local, track_type="local", color_by="global_id")
        mcmot.draw_traj(cams, ax_global, track_type="global", color_by="global_id")
        ax_local.show(pause=0.01)
        # ax_global.show(pause=0.01)

        cv2.imshow("out", img)
        cv2.waitKey(0)

    ax_local.show(True)
    # ax_global.show(True)


def main():
    # _test_show_images()

    _test_few_target_track()
    # _test_local_vs_global_id()
    # _test_sim_traj()

    """
    for camera_idx, camera_node in enumerate(camera_nodes):
        sample = data[0]
        camera_node.add_det(sample["boxes"][camera_idx])
        # p3d = camera_node.deproject_boxes()
    """
    # mcmot.cams[0].print_traj_state(start_time=0, end_time=50)


    """
    ax.clear()
    mcmot.draw_traj(ax, track_type="global", color_by="global_id")
    ax.show(block=True)
    """


if __name__ == "__main__":
    main()
