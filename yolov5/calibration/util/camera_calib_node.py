from math import atan
import numpy as np
import cv2
import yaml


from calibration.util.util import find_chessboard_corners, create_obj_points, \
    show_chessboard_corners, \
    draw_grid, draw_pose


def normalize(v):
    return v / np.linalg.norm(v, ord=1, axis=-1, keepdims=True)


def _draw_transform(img, transform_mat, scale, offset, color, arr_color, old_point=None, pos_z=False):
    al = scale
    radius = scale // 3

    p = transform_mat.dot(np.array([0, 0, 0, 1])) * scale
    d = normalize(transform_mat.dot(np.array([0, 0, -1 if not pos_z else 1, 0])))

    p1 = tuple((p[[0, 2]] + offset).astype(int))
    p2 = tuple((p[[0, 2]] + offset + al * d[[0, 2]]).astype(int))

    if old_point is not None:
        cv2.arrowedLine(img, old_point, p1, color=(120, 120, 120), thickness=2)

    cv2.arrowedLine(img, p1, p2, color=arr_color, thickness=2)
    cv2.circle(img, p1, radius=radius, color=tuple(int(c) for c in color), thickness=-1)


def _batched_dot(a, b):
    return np.sum(a * b, axis=-1)


def ray_ray_intersect(ray1_origin, ray1_dir, ray2_origin, ray2_dir):
    co_planer_threshold = 0.7

    da = ray1_dir	# Unnormalized direction of the ray
    db = ray2_dir
    dc = ray2_origin - ray1_origin

    da_cross_db = np.cross(da, db)

    """
    if np.abs(dc.dot(da_cross_db)) >= co_planer_threshold: # Lines are not coplanar
        return False
    """
    
    s = _batched_dot(np.cross(dc, db), da_cross_db) / _batched_dot(da_cross_db, da_cross_db)  # .LengthSquared;

    return s


def compute_img_point_dist_to_ground(cam, p_img, p_img_ground):
    p_ground = cam.deproject_to_ground(p_img_ground[None])[0]

    ray_cam_origin, ray_cam_dir = cam.get_camera_rays(p_img[None])
    ray_cam_dir = ray_cam_dir[0]

    ray_ground_origin = p_ground
    ray_ground_dir = np.float64([0, 1, 0])

    s = ray_ray_intersect(ray_ground_origin, ray_ground_dir, ray_cam_origin, ray_cam_dir)

    return s


class CalibrationPose:
    def __init__(self, rvec=None, tvec=None):
        self.rvec = np.array(rvec, dtype=float).reshape(3, 1) if rvec is not None else cv2.Rodrigues(np.eye(3))[0]
        self.tvec = np.array(tvec, dtype=float).reshape(3, 1) if tvec is not None else np.array([0, 0, 0], dtype=float)

    def inverse(self):
        rot_mat, _ = cv2.Rodrigues(self.rvec)

        rot_mat_cam = rot_mat.T

        rvec, _ = cv2.Rodrigues(rot_mat_cam)
        tvec = np.dot(-rot_mat_cam, self.tvec.reshape(-1))

        return CalibrationPose(rvec, tvec)
    
    def get_rot_mat(self):
        rot_mat, _ = cv2.Rodrigues(self.rvec)
        return rot_mat

    def get_mat(self):
        return self._get_transform_mat_from_vecs(self.rvec, self.tvec)

    def get_opengl_cam_mat(self):
        mat = self._get_transform_mat_from_vecs(self.rvec, self.tvec)
        mat[:3, 0:2] *= -1
        return mat

    @staticmethod
    def _get_transform_mat_from_vecs(rvec, tvec):
        rot_mat, _ = cv2.Rodrigues(rvec)

        mat = np.zeros([4, 4])
        mat[:3, :3] = rot_mat
        mat[:3, 3] = tvec.reshape(-1)
        mat[3, 3] = 1

        return mat

    @staticmethod
    def _get_vecs_from_mat(mat):
        rot_mat = mat[:3, :3]
        tvec = mat[:3, 3]

        rvec, _ = cv2.Rodrigues(rot_mat)

        return rvec, tvec

    @staticmethod
    def combine(p1, p2):
        mat1 = CalibrationPose._get_transform_mat_from_vecs(p1.rvec, p1.tvec)
        mat2 = CalibrationPose._get_transform_mat_from_vecs(p2.rvec, p2.tvec)

        matr = np.dot(mat1, mat2)

        return CalibrationPose(*CalibrationPose._get_vecs_from_mat(matr))

    @staticmethod
    def combine_range(poses):
        if len(poses) == 1:
            return poses[0]

        pose = CalibrationPose.combine(poses[0], poses[1])

        for p in poses[2:]:
            pose = CalibrationPose.combine(pose, p)

        return pose

    def get_translation(self):
        return self.tvec.reshape(-1)

    def transform(self, p):
        mat = self.get_mat()

        return mat.dot([*p, 1])[:3]

    @staticmethod
    def from_rot_mat_trans_vec(rotation_mat, translation_vec):
        rvec, _ = cv2.Rodrigues(rotation_mat)
        tvec = np.array(translation_vec).reshape(3, 1)

        return CalibrationPose(rvec, tvec)

    # accepts a scipy rotation
    @staticmethod
    def from_rot_trans(rotation, translation_vec):
        return CalibrationPose.from_rot_mat_trans_vec(rotation.as_matrix(), translation_vec)

    def set_euler(self, euler, degrees=False):
        self.rvec, _ = cv2.Rodrigues(Rotation.from_euler('xyz', euler, degrees=degrees).as_matrix())

    def get_state_dict(self):
        return {"rvec": self.rvec.tolist(), "tvec": self.tvec.tolist()}

    """
    def save(self, file):
        with open(file, "w") as f:
            json.dump(self.get_state_dict(), f)
    """

    @staticmethod
    def load(file):
        with open(file, "r") as f:
            return CalibrationPose.from_state_dict(json.load(f))

    @staticmethod
    def from_state_dict(dic):
        return CalibrationPose(np.array(dic["rvec"]), np.array(dic["tvec"]))

    def plot(self, ax, scale=1, vis_plane="xz"):
        def plot_line(p, color=(0,0,0), close=False):
            p = np.array(p)

            if close:
                p = np.concatenate([p, p[:1]], axis=0)

            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color)

        o = self.tvec.reshape(-1)

        rot_mat = cv2.Rodrigues(self.rvec)[0]

        for i, v in enumerate(rot_mat.T):
            color = [0, 0, 0]
            color[i] = 1
            plot_line([o, o + v * scale], color)

        if vis_plane is not None:
            if vis_plane == "xz":
                x = rot_mat[:, 0]
                y = rot_mat[:, 2]
            elif vis_plane == "xy":
                x = rot_mat[:, 0]
                y = rot_mat[:, 1]
            else:
                assert False

            p = [o + i * scale * x + j * scale * y for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            plot_line(p, close=True)

        # plt.show()


class CameraCalibrationNode:
    """
    def __init__(self, intr_calib, chessboard_size, chessboard_img, chess_origin_align="tl", visualize_calib=False, name="camera"):
        if isinstance(chessboard_img, str):
            chessboard_img = cv2.imread(chessboard_img)

        self.corners = None
        self.K = intr_calib.K
        self.D = intr_calib.D  # if D is not None else np.zeros(4)
        self.calib_size = intr_calib.size
        self.chessboard_size = chessboard_size
        self.chessboard_img = chessboard_img
        self.chess_origin_align = chess_origin_align
        self.name = name

        self.obj_to_cam = None
        self.cam_to_obj = None

        self.is_fisheye = False
        self.square_scale = None

        self._calibrate(visualize=visualize_calib)
    """

    def __init__(self, intr_calib=None, name="camera"):
        K, D, size = (intr_calib.K, intr_calib.D, tuple(intr_calib.size)) if intr_calib is not None else (None, None, None)

        self.corners = None
        self.K = K
        self.D = D  # if D is not None else np.zeros(4)
        self.calib_size = size
        self.chessboard_size = None
        self.chessboard_img = None
        self.chess_origin_align = None
        self.name = name

        self.obj_to_cam = None
        self.cam_to_obj = None

        self.is_fisheye = False
        self.square_scale = 1

        self.undistort_maps = None
        self.distorted_KD = None

    def is_undistorted(self):
        return self.undistort_maps is not None

    def scale_to(self, target_size):
        is_undist = self.is_undistorted()

        if is_undist:
            self.distort_camera()

        (w1, h1), (w2, h2) = self.calib_size, target_size

        assert w1 * h2 == w2 * h1, f"Aspect ratio does not match {w1 / h1} == {w2 / h2}"

        scale = w2 / w1

        self.K[:2] *= scale
        self.calib_size = (w2, h2)
        
        if is_undist:
            self.undistort_camera()

    def distort_camera(self):
        if self.undistort_maps is not None:
            self.K, self.D = self.distorted_KD
            self.distorted_KD = None
            self.undistort_maps = None

    def undistort_camera(self):
        if self.undistort_maps is None:
            map_type = cv2.CV_16SC2
            # map_type = cv2.CV_32FC1
            # map_type = cv2.CV_32FC2

            # create undistort map, modify K to ideal, set D to 0
            # -> proj/deprof now in undistorted space
            K_new, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, self.calib_size, 0, self.calib_size)
            # K_new = self.K
            map1, map2 = cv2.initUndistortRectifyMap(self.K, self.D, np.eye(3), K_new, self.calib_size, map_type)

            self.distorted_KD = self.K, self.D
            self.K = K_new
            self.D = np.zeros_like(self.D)
            self.undistort_maps = map1, map2

    def undistort_image(self, img):
        undistorted_img = cv2.remap(img, self.undistort_maps[0], self.undistort_maps[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    def process_image(self, img):
        assert img.shape[:2] == (self.calib_size[1], self.calib_size[0])

        if self.undistort_maps is not None:
            img = self.undistort_image(img)
        
        return img

    def save(self, file):
        assert self.undistort_maps is None, "distort camera before saving"

        state_dict = {
            "intr": {
                "K": self.K.tolist(),
                "D": self.D.tolist(),
                "calib_size": list(self.calib_size),
            },
            "extr": {
                **self.obj_to_cam.get_state_dict()
            } if self.obj_to_cam is not None else None
        }

        with open(file, "w") as f:
            yaml.dump(state_dict, f)

    @staticmethod
    def from_file(file, undist=False):
        node = CameraCalibrationNode()
        node.load(file)

        if undist:
            node.undistort_camera()

        return node

    def load(self, file):
        with open(file, "r") as f:
            state_dict = yaml.safe_load(f)

        self.K = np.array(state_dict["intr"]["K"])
        self.D = np.array(state_dict["intr"]["D"])
        self.calib_size = state_dict["intr"]["calib_size"]
    
        calib_pose = CalibrationPose.from_state_dict(state_dict["extr"]) if state_dict["extr"] is not None else None

        if calib_pose is not None:
            self.obj_to_cam = calib_pose
            self.cam_to_obj = self.obj_to_cam.inverse()

    def set_pose_from_rvec_tvec(self, rvec, tvec):
        self.obj_to_cam = CalibrationPose(rvec, tvec)
        self.cam_to_obj = self.obj_to_cam.inverse()

    def calibrate_from_points(self, obj_points, img_points, visualize_img=None):
        rvec, tvec = self._compute_camera_pose_from_points(obj_points, img_points)
        self.obj_to_cam = CalibrationPose(rvec, tvec)
        self.cam_to_obj = self.obj_to_cam.inverse()

        if visualize_img is not None:
            self.chessboard_img = visualize_img
            self.visualize_pose()

    def calibrate(self, chessboard_size, chessboard_img, chess_origin_align="tl", visualize_calib=False):
        self.chessboard_size = chessboard_size
        self.chessboard_img = chessboard_img
        self.chess_origin_align = chess_origin_align

        self._calibrate(visualize=visualize_calib)

    """
    def get_board_transform(self):
        return self.obj_to_cam

    def get_cam_transform(self):
        return self.obj_to_cam.inverse()
    """

    def get_fovy(self):
        fy = self.K[1, 1]
        w, h = self.calib_size
        return 2 * atan(h / (2 * fy))

    def set_intrinsics(self, K, D=None):
        if D is None:
            D = np.zeros(4)

        self.K = K
        self.D = D

    def draw_pose(self, img, point, *args, **kwargs):
        return draw_pose(
            img, self.obj_to_cam.rvec, self.obj_to_cam.tvec, self.K, self.D,
            is_fisheye=self.is_fisheye, point=point, *args, **kwargs)

    def visualize_coord_grid(self, img, size=10, scale=1):
        size2 = size + size - 1 
        obj_points = np.zeros((size2*size2, 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:size2, 0:size2].T.astype(np.float32).reshape(-1, 2)
        obj_points[:, :2] -= size - 1 
        img_points = self.project_points(obj_points * scale).astype(int)

        for (ox, oy, oz), (x, y) in zip(obj_points.astype(int), img_points):
            if img.ndim == 2:
                img = img[:, :, None].repeat(3, 2)

            cv2.circle(img, (x, y), 3, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, f"{ox},{oy}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 127, 127), 2)

        return img

        # cv2.imshow("calib", img)
        # cv2.waitKey(0)

    def visualize_pose(self, point=None, scale=1):
        scale *= self.square_scale
        img = draw_pose(
            np.array(self.chessboard_img), self.obj_to_cam.rvec, self.obj_to_cam.tvec, self.K, self.D,
            is_fisheye=self.is_fisheye, point=point, scale=scale)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    def _compute_camera_pose_from_points(self, obj_points, img_points):
        # not working for fisheye
        assert obj_points.ndim == 2
        assert img_points.ndim == 2

        assert obj_points.shape[0] == img_points.shape[0]
        assert obj_points.shape[1] == 3
        assert img_points.shape[1] == 2

        _, rvecs, tvecs = cv2.solvePnP(obj_points, img_points, self.K.astype(np.float32), self.D.astype(np.float32))

        return rvecs, tvecs

    def _detect(self, visualize=False):
        self.corners = find_chessboard_corners(
            self.chessboard_img, self.chessboard_size,
            origin_adjust=self.chess_origin_align)

        if visualize and self.corners is not None:
            show_chessboard_corners(self.chessboard_img, self.chessboard_size, self.corners)

        return self.corners is not None

    def undistort_points(self, points):
        if len(points.shape) == 1:
            points = points[None, None]
        elif len(points.shape) == 2:
            points = points[:, None]

        # points = cv2.fisheye.undistortPoints(points, K, D)
        points = cv2.undistortPoints(points, self.K, self.D)

        return points

    # naive with loop, slow
    def get_camera_rays2(self, points):
        if len(points.shape) == 1:
            points = points[None, None]

        # points = cv2.fisheye.undistortPoints(points, K, D)
        points = cv2.undistortPoints(points, self.K, self.D)

        rays = []

        mat = self.cam_to_obj.get_mat()
        ray_origin = mat[:3, 3]

        for point in points[:, 0]:
            # ray_origin = cam_mat[:3, 3]
            ray_dir = mat[:3, :3].dot(normalize(np.array([*point, 1])))
            rays.append(ray_dir)

        return ray_origin, np.array(rays)

    # returns the ray directions through the specified pixels
    def get_camera_rays(self, points):
        if len(points.shape) == 1:
            points = points[None, None]
        elif len(points.shape) == 2:
            points = points[:, None]

        # points = cv2.fisheye.undistortPoints(points, K, D)
        points = cv2.undistortPoints(points, self.K, self.D)

        # same, we use the transpose to simplify the dot product in numpy
        # mat_t = self.cam_to_obj.get_mat().T
        mat_t = self.obj_to_cam.get_mat()[:3, :3]
        ray_origin = self.cam_to_obj.get_mat()[:3, 3]  # camera position

        points = points[:, 0]
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

        ray_dirs = points / np.linalg.norm(points, axis=1, keepdims=True)
        ray_dirs = np.dot(ray_dirs, mat_t)
        # ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=1, keepdims=True)

        # ray_dirs[:, 1:] *= -1

        return ray_origin, ray_dirs

    def deproject_to_plane(self, points, plane_point, plane_normal):
        proj = points - np.dot(points - plane_point, plane_normal) * plane_normal
        off = proj - plane_point
        right = np.cross(off, np.array([0, 1, 0]))
        up = np.cross(right, off)

        return np.array([right, up])

    def deproject_to_ground(self, points, up_axis="z", omit_up_coord=False):
        ground_axes = [
            [1, 2],
            [0, 2],
            [0, 1]
        ]

        up_axis_id = "xyz".index(up_axis)
        ground_axis_id = ground_axes[up_axis_id]

        ray_origin, ray_dirs = self.get_camera_rays(points)
        depth = -(ray_origin[up_axis_id] / ray_dirs[:, up_axis_id])  # up axis is negative z

        if omit_up_coord:
            p = ray_origin[..., ground_axis_id] + depth[:, None] * ray_dirs[..., ground_axis_id]
        else:
            p = ray_origin + depth[:, None] * ray_dirs

        return p, depth

    def deproject_vert_lines_to_ground(self, points_bot, lengths, up_axis="z"):
        assert up_axis == "z"

        """
        ground_axes = [
            [1, 2],
            [0, 2],
            [0, 1]
        ]

        up_axis_id = "xyz".index(up_axis)
        ground_axis_id = ground_axes[up_axis_id]
        """

        p_ground, _ = self.deproject_to_ground(points_bot, up_axis=up_axis, omit_up_coord=False)

        points_top = points_bot.copy()
        points_top[:, 1] -= lengths
        ray_cam_origin, ray_cam_dir = self.get_camera_rays(points_top)

        ray_ground_dir = np.float64([0, 0, -1])  # up axis is negative z

        # s = np.array([ray_ray_intersect(o, ray_ground_dir, ray_cam_origin, d) for o, d in zip(p_ground, ray_cam_dir)])
        s = ray_ray_intersect(p_ground, ray_ground_dir[None], ray_cam_origin[None], ray_cam_dir)

        # out = ray_cam_origin + s[:, None] * ray_cam_dir
        out = p_ground
        out[:, 2] = s  # add height

        return out


    def deproject_points(self, img_points, depth):
        points = np.float32(img_points)
        depth = np.float32(depth)


        ray_origin, ray_dirs = self.get_camera_rays(points)

        p = ray_origin + depth * ray_dirs

        return p

    def project_points(self, points):
        K, D = self.K, self.D
        rvecs, tvecs = self.obj_to_cam.rvec, self.obj_to_cam.tvec

        if not self.is_fisheye:
            imgpts, jac = cv2.projectPoints(points[None], rvecs, tvecs, K, D)
            imgpts = imgpts[:, 0]
        else:
            imgpts, jac = cv2.fisheye.projectPoints(points[None], rvecs, tvecs, K, D)
            assert False
            imgpts = imgpts[0]

        return imgpts

    def visualize_points(self, points):
        pass

    def _calibrate(self, visualize=False, square_scale=1, point_offset=None, neg_z=False):
        self._detect()

        self.square_scale = square_scale

        obj_points = create_obj_points(self.chessboard_size, flip_yz=True)

        if neg_z:
            obj_points[:, :, 2] = self.chessboard_size[1] - 1 - obj_points[:, :, 2]

        if point_offset is not None:
            obj_points[0] += np.array(point_offset)

        obj_points *= square_scale

        rvec, tvec = self._compute_camera_pose_from_points(obj_points, self.corners)
        self.obj_to_cam = CalibrationPose(rvec, tvec)
        self.cam_to_obj = self.obj_to_cam.inverse()

        if visualize:
            self.visualize_pose()

        """
        pprint(self.obj_to_cam_l.get_mat())
        pprint(self.obj_to_cam_r.get_mat())
        pprint(self.board_l_to_board_r.get_mat())
        """

    def visualize(self, img):
        scale = 20
        size = (640, 640)
        # offset = np.array(size) * 0.3  # - center[[0, 2]] * scale
        offset = np.array(size) * 0.5

        board_offset = offset + (np.array(self.chessboard_size) - np.array((1, 1))) * 0.5 * scale

        if img is None:
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            draw_grid(img, scale * 2)

        _draw_transform(img, np.eye(4),
                        scale, offset=board_offset, color=(255, 0, 0),
                        arr_color=(255, 0, 0), pos_z=False)

        _draw_transform(img, self.cam_to_obj.get_mat(),
                        scale, offset=offset, color=(0, 255, 0),
                        arr_color=(0, 255, 0), pos_z=True)

        return img
