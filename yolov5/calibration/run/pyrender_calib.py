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

from util.util import calib_intrinsics_from_image_folder, draw_pose
from util.video_record import OpenCVVideoRecord


class _Viewer:
    def __init__(self, fovy, cam_pose=None) -> None:
        chessboard_img = cv2.imread("../data/camera_calib/sample/extrinsic/image_6.jpg")

        scene, mesh_node = self._setup_scene()
        cam = self._calib(chessboard_img)
        r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

        p = np.array([0.0, 2.0, 0.0])

        video_writer = OpenCVVideoRecord("out/out.webm", fps=30, start_rec=True)

        for _ in range(100):
            p_ground = p.copy()
            p_ground[1] = 0

            dist = p[1]

            img_cam = chessboard_img.copy()
            cam.draw_pose(img_cam, p_ground, scale=[1, dist, 1])

            pose = np.diag([5.0, 5.0, 5.0, 1.0])
            pose[:2, 3] = [-(p_ground[0] - 1.0) / 5, (p_ground[2] - 1.0) / 5]

            scene.set_pose(mesh_node, pose)

            img_render, _ = r.render(scene)

            img = np.concatenate([img_cam, img_render], 1)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            video_writer.update(img)

            cv2.imshow("out", img)
            cv2.waitKey(15)

            # p = p + np.random.uniform([-1, -1, -1], [1, 1, 1]) * 0.3
            p = p + np.array([1, 0, 1]) * 0.06

        video_writer.close()

    def _setup_scene(self, cam_pose=None):
        scene = pyrender.Scene()

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
        camera = pyrender.PerspectiveCamera(yfov=0.86)
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

        light = pyrender.PointLight(color=np.ones(3), intensity=450.0)
        scene.add(light, pose=light_pose)
        # r = pyrender.OffscreenRenderer(400, 400)

        return scene, mesh_node

    def _calib(self, chessboard_img):
        intr_calib = calib_intrinsics_from_image_folder("../data/camera_calib/sample/intrinsic", chessboard_size=(9, 6))

        cam = CameraCalibrationNode(intr_calib, chessboard_size=(9,6), chessboard_img=chessboard_img, visualize_calib=False)

        return cam

        # p_test = np.random.uniform([0, 0, 0], [5, 5, 5])
        p_test = np.array([0.0, 1.0, 0.0])
        p_ground_test = p_test.copy()
        p_ground_test[1] = 0

        ground_point, top_point = cam.project_points(np.stack([p_ground_test, p_test]))
        dist = compute_img_point_dist_to_ground(cam, top_point, ground_point)

    def _update_thread(self):
        

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
