import glob
import cv2
import numpy as np


def image_iter(idx):
    files = sorted(glob.glob(f"../data/Wildtrack_dataset/Image_subsets/C{idx+1}/*"))

    for file in files:
        yield cv2.imread(file)


def video_iter(idx):
    vid_file = f"../data/Wildtrack_dataset/vid/cam{idx+1}.mp4"

    cap = cv2.VideoCapture(vid_file)

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break

        yield img

    cap.close()


def main():
    for vid_idx in range(7):
        for img in image_iter(vid_idx):
            for frame_idx, frame in enumerate(video_iter(vid_idx)):
                if np.array_equal(img, frame):
                    break
        return


if __name__ == "__main__":
    main()
