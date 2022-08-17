import argparse
import yaml
import cv2
import os
from calibration.util.camera_calib_node import CameraCalibrationNode
from calibration.util.util import calib_intrinsics_from_image_folder, calib_intrinsics_from_video

from utils.mot.video_input import VideoInput, OpenCVVideoRecord


def extract_vid_frames(calib_video_path, count, out_path=None):
    if out_path is not None:
        os.makedirs(out_path)

    cap = cv2.VideoCapture(calib_video_path)

    vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    imgs = []

    keyframes = [round((i / (count - 1)) * (vid_length - 1)) for i in range(count)]

    for i in range(vid_length):
        # next_frame = round((i / (count - 1)) * (vid_length - 1))
        # cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        cap.grab()

        if i in keyframes:
            ret, frame = cap.retrieve()
            assert ret

            if out_path is not None:
                cv2.imwrite(os.path.join(out_path, f"{i:02d}.png"), frame)

            imgs.append(frame)

    return imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--size", type=int, nargs=2, required=True)
    parser.add_argument("--chess-size", type=int, nargs=2, required=True)
    parser.add_argument("--override-calib", action="store_true")
    parser.add_argument("--override-video", action="store_true")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    calib_img_count = 60
    cam_cfg = cfg["mot"]["cameras"][args.idx]

    if args.size is not None:
        cam_cfg["input"]["size"] = args.size

    calib_size = cam_cfg["input"]["size"]
    calib_file = cam_cfg["calib"]

    if not args.override_calib and os.path.isfile(calib_file):
        print(f"Calib '{calib_file}' already exists")
        return

    # TODO: append img size to filename
    assert calib_file.startswith("cfg/calib")
    calib_out_dir = os.path.join(calib_file.replace("cfg/calib", "cam_calib_data").replace(".yaml", ""), f"{calib_size[0]}_{calib_size[1]}")
    calib_img_dir = os.path.join(calib_out_dir, "intr_img")
    calib_video_file = os.path.join(calib_out_dir, "intr.avi")  # os.path.join("cam_calib_data", "intr", f"{args.idx}.mp4")

    if not args.override_video and (os.path.isfile(calib_video_file) or os.path.isfile(calib_img_dir)):
        if not os.path.isdir(calib_img_dir):
            print(f"Calibration video '{calib_video_file}' exists")
            extract_vid_frames(calib_video_file, calib_img_count, calib_img_dir)
        else:
            print(f"Calibration images '{calib_img_dir}' exists")
    else:
        assert not os.path.isdir(calib_img_dir), f"Delete {calib_img_dir} first"

        video_input = VideoInput(**cam_cfg["input"])
        video_output = OpenCVVideoRecord(calib_video_file, fps=30)

        print("Press R to record calibration video")

        while video_input.is_open():
            img = video_input()

            cv2.imshow("out", img)
            key = cv2.waitKey(1)

            video_output.update(img, key)

            if video_output.record_completed:
                break

        assert video_output.record_completed, "Video recording failed"

        print(f"Recorded video to {calib_video_file}")

        extract_vid_frames(calib_video_file, calib_img_count, calib_img_dir)
        return

    intr_calib = calib_intrinsics_from_image_folder(calib_img_dir, chessboard_size=args.chess_size, calib_size=calib_size,
        visualize=True, undistort_test=True)

    # assert intr_calib.size == cam_cfg["input"]["size"]
    calib_node = CameraCalibrationNode(intr_calib)

    os.makedirs(os.path.dirname(calib_file), exist_ok=True)
    calib_node.save(calib_file)


if __name__ == "__main__":
    main()
