from turtle import update
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import time
import threading
import cv2
from run.extrinsic_calib_height_meas import _draw_points, compute_img_point_dist_to_ground
from util.camera_calib_node import CameraCalibrationNode

from util.util import calib_intrinsics_from_image_folder


class _Viewer:
    def __init__(self, fovy, cam_pose=None) -> None:
        scene = pyrender.Scene()

        self.v = None

        fuze_trimesh = trimesh.load('../data/mesh/pyrender_examples/models/fuze.obj')
        # obj_trimesh = trimesh.primitives.Cylinder(radius=1, height=1)
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        plane_mesh = pyrender.Mesh.from_trimesh(trimesh.creation.box(extents=(5, 5, 0.01)))

        plane_pose = np.array([
            [1.0,  0.0, 0.0, 0],
            [0.0,  1.0, 0.0, 0],
            [0.0,  0.0, 1.0, 0.0],
            [0.0,  0.0, 0.0, 1.0],
        ])

        plane_node = scene.add(plane_mesh, pose=plane_pose)

        mesh_pose = np.array([
            [5.0,  0.0, 0.0, 0.0],
            [0.0,  5.0, 0.0, 0.0],
            [0.0,  0.0, 5.0, 0.0],
            [0.0,  0.0, 0.0, 1.0],
        ])

        mesh_node = scene.add(mesh, pose=mesh_pose)
        camera = pyrender.PerspectiveCamera(yfov=fovy)
        if cam_pose is None:
            s = np.sqrt(2)/2
            cam_pose = np.array([
            [0.0, -s,   s,   5.0],
            [1.0,  0.0, 0.0, 0.0],
            [0.0,  s,   s,   5.0],
            [0.0,  0.0, 0.0, 1.0],
            ])
        scene.add(camera, pose=cam_pose)

        light_pose = np.array([
            [1.0,  0.0, 0.0, 5],
            [0.0,  1.0, 0.0, 0],
            [0.0,  0.0, 1.0, 5.0],
            [0.0,  0.0, 0.0, 1.0],
        ])

        light = pyrender.PointLight(color=np.ones(3), intensity=150.0)
        scene.add(light, pose=light_pose)
        # r = pyrender.OffscreenRenderer(400, 400)

        v = pyrender.Viewer(scene, auto_start=False, viewport_size=(640, 480))

        self.scene = scene
        self.mesh_node = mesh_node
        self.v = v

        t = threading.Thread(target=self._update_thread)
        t.start()

        v.start()
        t.join()

    def _update_thread(self):
        cv2.namedWindow("out", cv2.WINDOW_NORMAL)

        chessboard_img = cv2.imread("../data/camera_calib/sample/extrinsic/image_6.jpg")
        intr_calib = calib_intrinsics_from_image_folder("../data/camera_calib/sample/intrinsic", chessboard_size=(9, 6))

        cam = CameraCalibrationNode(intr_calib, chessboard_size=(9,6), chessboard_img=chessboard_img, visualize_calib=False)

        # p_test = np.random.uniform([0, 0, 0], [5, 5, 5])
        p_test = np.array([0.0, 1.0, 0.0])
        p_ground_test = p_test.copy()
        p_ground_test[1] = 0

        ground_point, top_point = cam.project_points(np.stack([p_ground_test, p_test]))
        dist = compute_img_point_dist_to_ground(cam, top_point, ground_point)

        _draw_points(chessboard_img, [ground_point, top_point])

        i = 0
        while True:
            pose = np.eye(4)
            pose[:3,3] = [0, i, 0]
            self.v.render_lock.acquire()
            # self.scene.set_pose(self.mesh_node, pose)
            self.v.render_lock.release()
            i += 0.001
            
            cv2.imshow("out", chessboard_img)
            cv2.waitKey(100)


def main():
    """
    chessboard_img = cv2.imread("../data/camera_calib/sample/extrinsic/image_6.jpg")
    intr_calib = calib_intrinsics_from_image_folder("../data/camera_calib/sample/intrinsic", chessboard_size=(9, 6))

    cam = CameraCalibrationNode(intr_calib, chessboard_size=(9,6), chessboard_img=chessboard_img, visualize_calib=False)

    # p_test = np.random.uniform([0, 0, 0], [5, 5, 5])
    p_test = np.array([0.0, 1.0, 0.0])
    p_ground_test = p_test.copy()
    p_ground_test[1] = 0

    # p_ground_test, p_test = np.float64([0, 0, 0]), np.float64([0, 1.5, 0])

    ground_point, top_point = cam.project_points(np.stack([p_ground_test, p_test]))
    dist = compute_img_point_dist_to_ground(cam, top_point, ground_point)

    assert np.allclose(dist, p_test[1])  #  - p_ground_test[1]

    _draw_points(chessboard_img, [ground_point, top_point])

    cam_pose = cam.obj_to_cam.get_mat()
    """

    # cv2.imshow("out", chessboard_img)
    # cv2.waitKey(1)

    _Viewer(fovy=0.86, cam_pose=None)


if __name__ == "__main__":
    main()
