import argparse
import yaml
import cv2
import numpy as np
import os
from calibration.util.camera_calib_node import CameraCalibrationNode
from calibration.util.util import calib_intrinsics_from_image_folder, calib_intrinsics_from_video, find_calibration_points

from utils.mot.video_input import VideoInput, OpenCVVideoRecord


def main():
    intr_calib = calib_intrinsics_from_image_folder("../data/camera_calib/opencv_example/", chessboard_size=(9,6), visualize=False, undistort_test=False)

    calib_node = CameraCalibrationNode(intr_calib)
    calib_node.save("out/test_calib.yaml")

    test_img = cv2.imread("../data/camera_calib/opencv_example/left12.jpg")

    chessboard_size = (9,6)

    # 1
    _, p1, _ = find_calibration_points(
        chessboard_size, [test_img], visualize=True)
    p1 = p1[0]

    p1_undist = calib_node.undistort_points(p1)

    # 2
    calib_node.undistort_camera()
    
    test_img_undistort = calib_node.undistort_image(test_img)

    _, p2, _ = find_calibration_points(
        chessboard_size, [test_img_undistort], visualize=True)
    p2 = p2[0]

    p2_undist = calib_node.undistort_points(p2)

    p1_undist = p1_undist[:, 0]
    p2_undist = p2_undist[:, 0]

    diff = np.sqrt(np.sum((p1_undist - p2_undist) * (p1_undist - p2_undist), 1))

    print(diff)


if __name__ == "__main__":
    main()
