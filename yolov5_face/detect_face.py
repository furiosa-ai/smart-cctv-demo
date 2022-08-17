# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import numpy as np
import pickle

import os
import cv2
import torch
from numpy import random
import copy

from face_predictor.face_predictor import Yolov5FacePredictor


def show_results(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='data/images/test.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', required=True)
    opt = parser.parse_args()

    img = cv2.cvtColor(cv2.imread(opt.image), cv2.COLOR_BGR2RGB)
    model = Yolov5FacePredictor(opt.weights, input_format="chw", input_prec="f32")
    model.to(opt.device)
    out = model(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with open("result.pkl", "wb") as f:
        pickle.dump(out, f)

    for xyxy, conf, landmarks in out:
        img = show_results(img, xyxy, conf, landmarks, None)

    cv2.imwrite('result.jpg', img)


if __name__ == '__main__':
    main()
