import cv2
import numpy as np
from util.camera_calib_node import CameraCalibrationNode
from util.util import calib_intrinsics_from_image_folder


def _draw_points(img, points):
    for x, y in np.array(points).astype(int):
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)


def ray_ray_intersect(ray1_origin, ray1_dir, ray2_origin, ray2_dir):
    co_planer_threshold = 0.7

    da = ray1_dir	# Unnormalized direction of the ray
    db = ray2_dir
    dc = ray2_origin - ray1_origin

    da_cross_db = np.cross(da, db)

    if np.abs(dc.dot(da_cross_db)) >= co_planer_threshold: # Lines are not coplanar
        return False
    
    s = np.cross(dc, db).dot(da_cross_db) / np.dot(da_cross_db, da_cross_db)  # .LengthSquared;

    return s

    """
    if s >= 0.0 and s <= 1.0:	# Means we have an intersection
        intersection = ray1_origin + s * da

        # See if this lies on the segment
        if (intersection - segment.Start).LengthSquared + (intersection - segment.End).LengthSquared <= segment.LengthSquared + lengthErrorThreshold:
            return True

    return False
    """


def compute_img_point_dist_to_ground(cam, p_img, p_img_ground):
    p_ground = cam.deproject_to_ground(p_img_ground[None])[0]

    ray_cam_origin, ray_cam_dir = cam.get_camera_rays(p_img[None])
    ray_cam_dir = ray_cam_dir[0]

    ray_ground_origin = p_ground
    ray_ground_dir = np.float64([0, 1, 0])

    s = ray_ray_intersect(ray_ground_origin, ray_ground_dir, ray_cam_origin, ray_cam_dir)

    return s


def main():
    chessboard_img = cv2.imread("../data/camera_calib/sample/extrinsic/image_6.jpg")
    intr_calib = calib_intrinsics_from_image_folder("../data/camera_calib/sample/intrinsic", chessboard_size=(9, 6))

    cam = CameraCalibrationNode(intr_calib, chessboard_size=(9,6), chessboard_img=chessboard_img, visualize_calib=False)

    p_test = np.random.uniform([0, 0, 0], [5, 5, 5])
    p_ground_test = p_test.copy()
    p_ground_test[1] = 0

    # p_ground_test, p_test = np.float64([0, 0, 0]), np.float64([0, 1.5, 0])

    ground_point, top_point = cam.project_points(np.stack([p_ground_test, p_test]))
    dist = compute_img_point_dist_to_ground(cam, top_point, ground_point)

    assert np.allclose(dist, p_test[1])  #  - p_ground_test[1]

    _draw_points(chessboard_img, [ground_point, top_point])

    cv2.imshow("out", chessboard_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
