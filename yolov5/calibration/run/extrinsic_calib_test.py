
from util.camera_calib_node import CameraCalibrationNode
from util.util import calib_intrinsics_from_image_folder


def main():
    intr_calib = calib_intrinsics_from_image_folder("../data/camera_calib/sample/intrinsic", chessboard_size=(9, 6))

    calib_node = CameraCalibrationNode(intr_calib, chessboard_size=(9,6), chessboard_img="../data/camera_calib/sample/extrinsic/image_6.jpg", visualize_calib=True)

    pass


if __name__ == "__main__":
    main()
