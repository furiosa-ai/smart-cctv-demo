from collections import namedtuple
import cv2
import glob
import os
import numpy as np
from tqdm import tqdm


IntrinsicCalibrationData = namedtuple("IntrinsicCalibrationData", ["K", "D", "size"])



def _extend_chessboard(corners, side=None):
    if side is None:
        for s in ["top", "bottom", "left", "right"]:
            corners = _extend_chessboard(corners, s)

        return corners

    if side == "top":
        strip = corners[:1] - (corners[1:2] - corners[:1])
        corners = np.concatenate([strip, corners], axis=0)
    elif side == "bottom":
        strip = corners[-1:] - (corners[-2:-1] - corners[-1:])
        corners = np.concatenate([corners, strip], axis=0)
    elif side == "left":
        strip = corners[:, :1] - (corners[:, 1:2] - corners[:, :1])
        corners = np.concatenate([strip, corners], axis=1)
    elif side == "right":
        strip = corners[:, -1:] - (corners[:, -2:-1] - corners[:, -1:])
        corners = np.concatenate([corners, strip], axis=1)

    return corners


def create_obj_points(chessboard_size, flip_yz=False):
    objp = np.zeros((1, chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    if flip_yz:
        objp = objp[:, :, [0, 2, 1]]  # flip y and z

    return objp


def undistort_image(img, K, D, DIM):
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, DIM, 0, DIM)
    # K_new = K
    map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K_new, tuple(DIM), cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img


def undistort_fisheye_image(img, K, D, DIM):
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, tuple(DIM), cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img


def adjust_corner_origin(corners, chessboard_size, origin_adjust):
    if origin_adjust is not None:
        x_lower = origin_adjust[1] == "l"
        y_lower = origin_adjust[0] == "t"

        corners = corners.reshape((chessboard_size[1], chessboard_size[0], *corners.shape[1:]))

        sx, sy = corners[0, 0, 0]
        ex, ey = corners[chessboard_size[1] - 1, chessboard_size[0] - 1, 0]

        # flip x
        if (sx - ex > 0) == x_lower:
            corners = np.flip(corners, 1)

        # flip y
        if (sy - ey > 0) == y_lower:
            corners = np.flip(corners, 0)


        corners = corners.reshape((chessboard_size[1] * chessboard_size[0], *corners.shape[2:]))

    return corners


def find_chessboard_corners(image, chessboard_size, origin_adjust=None, rescale=None):
    find_corner_flags = cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK

    if rescale is not None:
        ratio = rescale / image.shape[1]
        image = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))

    calib_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    org_img_gray = calib_img

    ret, corners = cv2.findChessboardCorners(
        org_img_gray,
        chessboard_size,
        flags=find_corner_flags
    )

    if ret:
        corners = adjust_corner_origin(corners, chessboard_size, origin_adjust)

        if rescale is not None:
            corners /= ratio

        return corners
    else:
        return None


def find_calibration_points(chessboard_size, calib_imgs, resize_size=None, visualize=False, flip_order=False):
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    find_corner_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

    obj_points = []
    img_points = []

    objp = np.zeros((1, chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    calib_img_size = None

    for calib_img in tqdm(calib_imgs, desc="Detecting corners"):
        if resize_size is not None:
            calib_img = cv2.resize(calib_img, resize_size)

        calib_img = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)

        _calib_img_size = (calib_img.shape[1], calib_img.shape[0])

        if calib_img_size is None:
            calib_img_size = _calib_img_size
        else:
            assert calib_img_size == _calib_img_size, "All images must share the same size."

        ret, corners = cv2.findChessboardCorners(
            calib_img,
            chessboard_size,
            flags=find_corner_flags
        )

        print(corners)

        if ret:
            obj_points.append(objp)
            # is inplace
            cv2.cornerSubPix(calib_img, corners, (11, 11), (-1, -1), subpix_criteria)

            if flip_order:
                corners = np.flip(corners, axis=0)

            img_points.append(corners)

            if visualize:
                calib_img = np.stack([calib_img] * 3, axis=2)
                cv2.drawChessboardCorners(calib_img, chessboard_size, corners, ret)
                cv2.imshow("calib", calib_img)
                cv2.waitKey(16)
        else:
            print(f"invalid.")
            # assert False

    return np.float32(obj_points), img_points, calib_img_size


def calib_intrinsics_from_images(calib_images, chessboard_size, is_fisheye=False, calib_size=None, visualize=False,
                                 calib_file=None, no_cache=True, undistort_test=False):
    if calib_file is not None and not no_cache and os.path.exists(calib_file):
        calib_data = np.load(calib_file)
        K, D, calib_img_size = calib_data["K"], calib_data["D"], calib_data["size"]
    else:
        if isinstance(calib_images[0], str):
            calib_images = [cv2.imread(f) for f in calib_images]

        if calib_size is not None:
            assert all(img.shape[:2] == (calib_size[1], calib_size[0]) for img in calib_images), \
            f"image size {[calib_images[0].shape[1], calib_images[0].shape[0]]}, Calibration size {calib_size}"

        calib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        obj_points, img_points, calib_img_size = find_calibration_points(
            chessboard_size, calib_images, resize_size=calib_size, visualize=visualize)

        calib_valid = len(obj_points)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(calib_valid)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(calib_valid)]

        print("Calibrating ...")
        if is_fisheye:
            fisheye_calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_chessboard_COND + cv2.fisheye.CALIB_FIX_SKEW
            rms, _, _, _, _ = cv2.fisheye.calibrate(
                obj_points,
                img_points,
                calib_img_size,
                K, 
                D,
                rvecs,
                tvecs,
                fisheye_calibration_flags,
                calib_criteria
            )
        else:
            # rms, _, _, _, _ = cv2.calibrateCamera(obj_points, img_points, calib_img_size, K, D)
            ret, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, calib_img_size, None, None)

        k = list(D.reshape(-1))

        f = [K[0, 0], K[1, 1]]
        c = [K[0, 2], K[1, 2]]

        print("Found " + str(calib_valid) + " valid images for calibration")
        print("DIM=" + str(calib_img_size))
        print(f"f = {f}")
        print(f"c = {c}")
        print(f"k = {k}")
        # print("D=np.array(" + str(D.tolist()) + ")")
        print()

        if not no_cache and calib_file is not None:
            np.savez(calib_file, K=K, D=D, size=calib_img_size)

    if undistort_test:
        if is_fisheye:
            cv2.imshow("undistort", undistort_fisheye_image(calib_images[0], K, D, calib_img_size))
            cv2.waitKey(0)
        else:
            test_img = calib_images[0]
            cv2.imshow("undistort", np.concatenate([undistort_image(test_img, K, D, calib_img_size)], 0))
            cv2.waitKey(0)

    return IntrinsicCalibrationData(K, D, calib_img_size)


def calib_intrinsics_from_video(calib_video_path, count, **kwargs):
    cap = cv2.VideoCapture(calib_video_path)

    vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    imgs = []

    for i in range(count):
        next_frame = round((i / (count - 1)) * (vid_length - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        ret, frame = cap.read()
        assert ret
        imgs.append(frame)

    return calib_intrinsics_from_images(imgs, **kwargs)


def calib_intrinsics_from_image_folder(image_dir, chessboard_size, is_fisheye=False, **kwargs):
    files = glob.glob(os.path.join(image_dir, "*.*"))
    # calib_images = [cv2.imread(f) for f in files]
    return calib_intrinsics_from_images(
        files, chessboard_size, is_fisheye, calib_file=os.path.join(image_dir, "calib_intrinsics.npz"),
        **kwargs
    )



def show_chessboard_corners(chessboard_img, chessboard_size, corners, rescale=None):
    chessboard_img = np.array(chessboard_img)

    if rescale is not None:
        ratio = rescale / chessboard_img.shape[1]
        chessboard_img = cv2.resize(chessboard_img, (int(chessboard_img.shape[1] * ratio), int(chessboard_img.shape[0] * ratio)))
        corners = corners * ratio

    vis_img = cv2.drawChessboardCorners(chessboard_img, chessboard_size, corners, True)

    cv2.imshow("corners", vis_img)
    return cv2.waitKey(0)


def draw_grid(image, size, shift_half=False):
    start_white = False

    extend = 0 if not shift_half else size // 2

    yy = range(-extend, image.shape[0] + extend, size)
    xx = range(-extend, image.shape[1] + extend, size)

    for y1, y2 in zip(yy[:-1], yy[1:]):
        start_white = not start_white
        is_white = start_white

        for x1, x2 in zip(xx[:-1], xx[1:]):
            points = np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]]).astype(int)

            color = (255, 255, 255) if is_white else (0, 0, 0)
            cv2.fillConvexPoly(image, points, color=color)
            is_white = not is_white

            cv2.fillConvexPoly(image, points, color=color)

    return image


def draw_chessboard(image, chessboard_size, corners, y_range=None, x_range=None):
    corners = corners.reshape((chessboard_size[1], chessboard_size[0], *corners.shape[1:]))

    if y_range is not None and x_range is not None:
        # corners = corners[np.array(y_range, dtype=int)[:, None], np.array(x_range, dtype=int)]
        corners = corners[y_range][:, x_range]

    corners = _extend_chessboard(corners)

    yy = list(range(corners.shape[0]))
    xx = list(range(corners.shape[1]))

    # i = 0

    start_white = False
    is_white = False

    for y1, y2 in zip(yy[:-1], yy[1:]):
        start_white = not start_white
        is_white = start_white

        for x1, x2 in zip(xx[:-1], xx[1:]):
            # points = corners[y:y+2, x:x+2, 0, :]
            points = np.array([corners[y1, x1], corners[y1, x2], corners[y2, x2], corners[y2, x1]]).astype(int)[:, 0]

            color = (255, 255, 255) if is_white else (0, 0, 0)
            # color = (i, i, i)
            # color = tuple(np.random.randint(0, 255, 3).astype(float))
            cv2.fillConvexPoly(image, points, color=color)
            is_white = not is_white


def draw_pose(img, rvecs, tvecs, K, D, is_fisheye=False, point=None, scale=1):
    axis = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * scale

    if point is not None:
        axis += np.array(point)

    if not is_fisheye:
        imgpts, jac = cv2.projectPoints(axis[None], rvecs, tvecs, K, D)
    else:
        imgpts, jac = cv2.fisheye.projectPoints(axis[None], rvecs, tvecs, K, D)
        imgpts = imgpts[0]

    imgpts = imgpts.round().astype(int)

    corner = tuple(imgpts[0].ravel())

    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 0, 255), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[3].ravel()), (255, 0, 0), 5)

    return img


class UserCornerSelect:
    cur_image_idx = 0

    def __init__(self, img, num_points, scale=1, conn_poly=True):
        self.img = img
        self.points = [(0, 0)]
        self.num_points = num_points
        self.scale = scale
        self.conn_poly = conn_poly
        self.img_draw = None

        cv2.namedWindow('select', cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback('select', self.mouse_callback)

        while True:
            self.draw()
            cv2.imshow("select", self.img_draw)
            key = cv2.waitKey(30)

            if key != -1:
                break

        if self.num_points is None or len(self.points) - 1 == self.num_points:
            if self.num_points is not None:
                self.points = self.points[:-1]

            self.points = np.float32(self.points)
            self.points /= self.scale
        else:
            self.points = None

    @staticmethod
    def select_corners(*args, **kwargs):
        uc = UserCornerSelect(*args, **kwargs)
        corners = uc.points
        return corners

    def get_points(self):
        return self.points

    def mouse_callback(self, event, x, y, flags, param):
        if len(self.points) - 1 == self.num_points:
            return

        self.points[-1] = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))

    def draw(self):
        def draw_cross(img, pos, size, color, thickness):
            x, y = pos

            cv2.line(img, (x - size, y), (x + size, y), color, thickness)
            cv2.line(img, (x, y - size), (x, y + size), color, thickness)

        img_draw = np.array(self.img)
        img_draw = cv2.resize(
            img_draw,
            (round(img_draw.shape[1] * self.scale), round(img_draw.shape[0] * self.scale)))

        line_points = self.points

        if self.conn_poly:
            cv2.line(img_draw, line_points[0], line_points[-1], color=(0, 0, 255), thickness=1)

        for p1, p2 in zip(line_points[:-1], line_points[1:]):
            cv2.line(img_draw, p1, p2, color=(0, 0, 255), thickness=1)

        for p in self.points:
            draw_cross(img_draw, p, size=5, color=(0, 0, 255), thickness=1)

        self.img_draw = img_draw
