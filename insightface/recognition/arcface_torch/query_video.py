
from ai_util.imp_env import ImpEnv
from ai_util.dataset import ImageDataset
import argparse
import time
import numpy as np
import torch
import os
import sys
import cv2
from pathlib import Path
import mxnet
import pandas

with ImpEnv("yolov5_face"):
    from yolov5_face.face_predictor.face_predictor import Yolov5FacePredictor

from recognition.arcface_torch.utils.arcface_predictor import ArcFacePredictor
from recognition.arcface_torch.utils.face_extract import FaceExtractor
from recognition.arcface_torch.utils.face_gallery import FaceGallery



def show_img(img):
    cv2.imshow("out", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def read_img(img_file):
    return cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB)


def write_img(img_file, img):
    return cv2.imwrite(str(img_file), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def query_demo(args):
    t1 = time.time()

    if args.device == "furiosa":
        det_dev = "furiosa:npu0pe0"
        reg_dev = "furiosa:npu0pe1"
        # reg_dev = "onnx"
    else:   
        det_dev, reg_dev = args.device, args.device

    if args.calib_mode is not None and ("_" in args.calib_mode):
        face_det_calib, feat_extr_calib = args.calib_mode.split("_")
    else:
        face_det_calib = args.calib_mode
        feat_extr_calib = args.calib_mode

    face_det = Yolov5FacePredictor(cfg="yolov5_face/models/yolov5m_relu.yaml", weights="yolov5_face/weights/my/yolov5_relu.pt", 
        input_format="hwc", input_prec="i8", calib_mode=face_det_calib, quant_tag=face_det_calib).to(det_dev)

    feat_extr = ArcFacePredictor(cfg="configs/ms1mv3_r50_leakyrelu.py", weights="runs/ms1mv3_r50_leakyrelu_1/model.pt", 
        input_format="hwc", input_prec="i8", calib_mode=feat_extr_calib, quant_tag=feat_extr_calib,
        normalize=True, batch_size=1, pad_batch=True).to(reg_dev)

    gallery_cache_name = os.path.abspath(args.gallery).replace("/", "_") + (f"_{args.calib_mode}" if args.calib_mode is not None else "")

    face_det.set_async_inference(True)
    feat_extr.set_async_inference(True)

    gallery = FaceGallery(
        name=gallery_cache_name, 
        data=ImageDataset(args.gallery, limit=None, frame_step=5), 
        data_extr=FaceExtractor(face_det), 
        feat_extr=feat_extr
    )

    face_det.set_async_inference(False)
    feat_extr.set_async_inference(False)

    print("Created gallery")
    
    query_db = FaceGallery(
        data=ImageDataset([args.query]), 
        data_extr=FaceExtractor(face_det, single_face=True), 
        feat_extr=feat_extr
    )

    query_results = gallery.query([query_db[0]["feat"]])[0]

    out_dir = Path("results")

    if args.calib_mode is not None:
        out_dir /= args.calib_mode

    out_dir.mkdir(exist_ok=True)

    gallery.data.open()

    write_img(out_dir / "query.jpg", query_db.vis_sample(0))
    for topi, res in enumerate(query_results):
        idx = res.entry_idx
        dist = res.dist

        write_img(out_dir / f"top_{topi+1}.jpg", gallery.vis_sample(idx, label=str(dist)))

    print(f"Took {time.time() - t1:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--calib_mode", default="entropy_minmax")
    args = parser.parse_args()

    query_demo(args)


if __name__ == "__main__":
    main()
