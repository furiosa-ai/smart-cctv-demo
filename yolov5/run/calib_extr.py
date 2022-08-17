import argparse
import yaml
import cv2
import os
import numpy as np
from calibration.util.camera_calib_node import CameraCalibrationNode
from calibration.util.util import UserCornerSelect, calib_intrinsics_from_video, find_calibration_points

from utils.mot.video_input import VideoInput


def _read_chess_region_from_cfg(cfg, idx):
    camera_chess_region = cfg["camera_chess_regions"][idx]
    square_scale = cfg["square_scale"]

    assert len(camera_chess_region) in [4, 6]

    x1, y1, w, h = camera_chess_region[:4]

    if len(camera_chess_region) == 6:
        assert square_scale == 1, "If number of squares is given, scale should be in meter"
        num_squares_x, num_squares_y = camera_chess_region[4:]
    else:
        assert isinstance(w, int)
        assert isinstance(h, int)
        num_squares_x, num_squares_y = w, h

    return x1, y1, w, h, num_squares_x, num_squares_y, square_scale


def create_object_points(cfg, idx):
    x1, y1, w, h, num_squares_x, num_squares_y, square_scale = _read_chess_region_from_cfg(cfg, idx)

    sx, sy = num_squares_x + 1, num_squares_y + 1

    objp = np.zeros((sx * sy, 3), np.float32)
    objp[:, :2] = np.mgrid[0:sx, 0:sy].T.reshape(-1, 2)

    objp[:, 0] = objp[:, 0] * (w / num_squares_x) + x1
    objp[:, 1] = objp[:, 1] * (h / num_squares_y) + y1
    objp *= square_scale

    print(objp)

    return objp


def find_image_points(cfg, idx, img, visualize=True):
    calib_mode = cfg["mode"]
    x1, y1, w, h, num_squares_x, num_squares_y, square_scale = _read_chess_region_from_cfg(cfg, idx)

    if calib_mode == "manual":
        assert num_squares_x == 1, f"num_squares_x ({num_squares_x}) == 1"
        assert num_squares_y == 1, f"num_squares_y ({num_squares_y}) == 1"

        img_points = UserCornerSelect.select_corners(img=img, num_points=4, scale=1)  # selecting clockwise starting at tl corner
        assert len(img_points) == 4
        img_points = img_points[[0, 1, 3, 2]]  # rearange
        # img_points = img_points[:, None]
    elif calib_mode in ["semi", "auto"]:
        if calib_mode == "semi":
            poly_points = UserCornerSelect.select_corners(img=img, num_points=None, scale=1).astype(np.int32)
            mask = np.ones_like(img[:, :, 0])
            cv2.fillPoly(mask, [poly_points], color=0)  # , lineType=cv2.FILLED
            masked_img = img.copy()
            masked_img[mask > 0] = 127
            cv2.imshow("out", masked_img); cv2.waitKey(0)
        else:
            masked_img = img

        _, (img_points,), _ = find_calibration_points((num_squares_x + 1, num_squares_y + 1), [masked_img], visualize=visualize)
        img_points = img_points[:, 0]
    else:
        raise Exception(calib_mode)

    print(img_points)
    return img_points


class ProjectionTest:
    def __init__(self, calib_node, img, scale=1):
        self.calib_node = calib_node
        self.bg_img = img

        self.imgp = np.zeros(2)
        self.scale = scale

        cv2.namedWindow('proj')
        cv2.setMouseCallback('proj', self.mouse_callback)

        while True:
            cv2.imshow("proj", self._draw())
            key = cv2.waitKey(30)

            if key != -1:
                break

    def mouse_callback(self, event, x, y, flags, param):
        self.imgp[:] = (x, y)

    def _draw(self):
        img = self.bg_img.copy()

        imgp = self.imgp
        objp = self.calib_node.deproject_to_ground(imgp, up_axis="z", omit_up_coord=True)[0][0]

        ox_m, oy_m = objp
        ox, oy = ox_m / self.scale, oy_m / self.scale
        cv2.putText(img, f"{ox:.2f},{oy:.2f} | meter: {ox_m:.2f},{oy_m:.2f}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return img


def main():
    # generate path for extr image automatically
    # shoot an image per camera and store it if it doesnt exist

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--extr-cfg", required=True)
    parser.add_argument("--size", type=int, nargs=2, required=True)
    parser.add_argument("--override-calib", action="store_true")
    parser.add_argument("--override-img", action="store_true")
    parser.add_argument("--undistort", action="store_true")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    with open(args.extr_cfg, "r") as f:
        extr_cfg = yaml.safe_load(f)

    cam_idx = args.idx

    cam_cfg = cfg["mot"]["cameras"][cam_idx]

    if args.size is not None:
        cam_cfg["input"]["size"] = args.size

    calib_file = cam_cfg["calib"]
    calib_size = cam_cfg["input"]["size"]

    assert os.path.isfile(calib_file), f"No (intrinsic) calib file '{calib_file}'"
    calib_node = CameraCalibrationNode.from_file(calib_file)

    if args.undistort:
        calib_node.undistort_camera()

    calib_node.scale_to(cam_cfg["input"]["size"])

    """
    if not args.override_calib and os.path.isfile(calib_file):
        print(f"Calib '{calib_file}' already exists")
        return
    """

    # calib_video_file = os.path.join("cam_calib_data", "intr", f"{args.idx}.mp4")
    assert calib_file.startswith("cfg/calib")
    calib_out_dir = os.path.join(calib_file.replace("cfg/calib", "cam_calib_data").replace(".yaml", ""), f"{calib_size[0]}_{calib_size[1]}")
    calib_extr_img_file = os.path.join(calib_out_dir, "extr_img.png")

    if not args.override_calib and calib_node.obj_to_cam is not None:
        print(f"Calib '{calib_file}' already has extr calib")
        extr_img = cv2.imread(calib_extr_img_file)
    else:
        if not args.override_img and calib_node.obj_to_cam is not None:
            print(f"Extrinsic calibration image '{calib_extr_img_file}' exists -> reusing")
            extr_img = cv2.imread(calib_extr_img_file)
        else:
            video_input = VideoInput(**cam_cfg["input"])

            print("Press R to record calibration screenshot")

            extr_img = None
            while video_input.is_open():
                img = video_input()
                img = calib_node.process_image(img)

                cv2.imshow("out", img)
                key = cv2.waitKey(1) & 0xff

                if key == ord("r"):
                    extr_img = img
                    break

            assert extr_img is not None, "Failed to capture frame"
            cv2.imwrite(calib_extr_img_file, extr_img)

            print(f"Saved image to {calib_extr_img_file}")

        assert [extr_img.shape[1], extr_img.shape[0]] == calib_size, f"image size {[extr_img.shape[1], extr_img.shape[0]]}, Calibration size {calib_size}"

        # calib_node.save(calib_file)

        obj_points = create_object_points(extr_cfg, cam_idx)
        img_points = find_image_points(extr_cfg, cam_idx, extr_img)

        calib_node.calibrate_from_points(obj_points, img_points, visualize_img=extr_img)

        calib_node.distort_camera()
        calib_node.save(calib_file)

    calib_node.undistort_camera()
    # extr_img = calib_node.process_image(extr_img) dont undistort twice
    img = calib_node.visualize_coord_grid(extr_img, scale=extr_cfg["square_scale"])
    ProjectionTest(calib_node, img, scale=extr_cfg["square_scale"])


if __name__ == "__main__":
    main()
