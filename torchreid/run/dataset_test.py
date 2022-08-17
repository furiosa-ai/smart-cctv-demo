import cv2
from pathlib import Path
import re
import numpy as np
import pickle

from utils.reid import PRWDataset, ReIdGallery, draw_boxes


def show_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("out", img)
    cv2.waitKey(0)


def demo1():
    data = PRWDataset("../data/PRW-v16.04.20")

    cam_idx, seq_idx = 0, 0

    for sample in data.iter_frames(cam_idx, seq_idx):
        img = sample["image"]
        box = sample["box"]
        draw_boxes(img, box)
        cv2.imshow("out", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)


def demo2():
    data = PRWDataset("../data/PRW-v16.04.20")
    gal = ReIdGallery()

    cam_idx, seq_idx = 0, 0

    gal.add_sequence("prw", data.iter_frames(cam_idx, seq_idx))


def main():
    demo2()


if __name__ == "__main__":
    main()
