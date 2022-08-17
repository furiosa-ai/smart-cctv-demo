

from util.util import calib_intrinsics_from_image_folder


def main():
    calib = calib_intrinsics_from_image_folder("../data/camera_calib/sample/intrinsic", chessboard_size=(9, 6))

    pass


if __name__ == "__main__":
    main()
