import cv2
import numpy as np
from util.axes_plot_3d import AxesPlot3d
from util.camera_calib_node import CameraCalibrationNode
from util.util import calib_intrinsics_from_image_folder, draw_pose


def main():
    chessboard_img = cv2.imread("../data/camera_calib/sample/extrinsic/image_6.jpg")
    intr_calib = calib_intrinsics_from_image_folder("../data/camera_calib/sample/intrinsic", chessboard_size=(9, 6))

    cam = CameraCalibrationNode(intr_calib)
    cam.calibrate(chessboard_size=(9,6), chessboard_img=chessboard_img, visualize_calib=False)
    ax = AxesPlot3d()

    point = np.array([0, 1, 0])
    point_dir = np.array([0.8, 0.0, 0.2])
    t = 0.3

    trail = []

    for _ in range(60):
        trail.append(point)

        ground_point = point.copy()
        ground_point[1] = 0

        img = chessboard_img.copy()
        img = cam.draw_pose(img, ground_point, scale=[1, point[1], 1])

        plot_x, plot_y, plot_z = np.array(trail).T
        ax.clear()
        ax.scatter(plot_x, plot_y, plot_z)
        ax.show()

        cv2.imshow("out", img)
        cv2.waitKey(0)

        point = point + np.random.uniform([-1, -1, -1], [1, 1, 1]) * t

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
