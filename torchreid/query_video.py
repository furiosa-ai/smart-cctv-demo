import sys
import os
from pathlib import Path

sys.path.insert(0, os.getcwd())

import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import onnx
import cv2
import pandas
from ai_util.dataset import ImageDataset
from ai_util.imp_env import ImpEnv

with ImpEnv("yolov5"):
    from yolov5.yolo_predictor.predictor import Yolov5Predictor
    from yolov5.yolo_predictor.tracker import Tracker

from utils.box_extract import BoxExtractor, BoxExtractorIdentity
from utils.reid import ReIdGallery
from utils.reid_predictor import ReIdPredictor


def show_img(img):
    cv2.imshow("out", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def read_img(img_file):
    return cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB)


def write_img(img_file, img):
    return cv2.imwrite(str(img_file), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def query_demo(args):
    if args.device == "furiosa":
        det_dev = "furiosa:npu0pe0"
        reg_dev = "furiosa:npu0pe1"
    else:
        det_dev, reg_dev = args.device, args.device

    if args.calib_mode is not None and "_" in args.calib_mode:
        person_det_calib, feat_extr_calib = args.calib_mode.split("_")
    else:
        person_det_calib = args.calib_mode
        feat_extr_calib = args.calib_mode

    person_det = Yolov5Predictor(cfg="yolov5/models/yolov5m_warboy.yaml", weights="../weights/yolov5m_warboy.pt", name="../yolov5m_warboy",
        calib_data="../data/CrowdHuman/images/val/*",
        input_format="hwc", input_prec="i8", calib_mode=person_det_calib, quant_tag=person_det_calib,
        input_size=(640, 640)).to(det_dev)

    feat_extr = ReIdPredictor(cfg="configs/im_r50_softmax_256x128_amsgrad.yaml", weights="../weights/resnet50_market_xent.pth.tar",
        name="../resnet50_market_xent",
        input_format="hwc", input_prec="i8", output_type="np", batch_size=1, pad_batch=True, 
        calib_mode=feat_extr_calib, quant_tag=feat_extr_calib,).to(reg_dev)

    person_det.set_async_inference(True)
    feat_extr.set_async_inference(True)

    tracker = None
    # tracker = Tracker()

    gallery = ReIdGallery(
        name="prw" + (f"_{args.calib_mode}" if args.calib_mode is not None else ""),
        gallery_dir="../galleries",
        data=ImageDataset(args.gallery, frame_step=10), 
        data_extr=BoxExtractor(person_det, tracker=tracker), 
        feat_extr=feat_extr
    )

    person_det.set_async_inference(False)
    feat_extr.set_async_inference(False)

    query_db = ReIdGallery(
        data=ImageDataset([args.query]),
        data_extr=BoxExtractorIdentity(), 
        # data_extr=BoxExtractor(person_det, single_box=True), 
        feat_extr=feat_extr
    )

    query_results = gallery.query([query_db[0]["feat"]])[0]

    out_dir = Path("../results/reid/")
    out_dir.mkdir(parents=True, exist_ok=True)

    """
    if args.calib_mode is not None:
        out_dir /= args.calib_mode
    
    """

    write_img(out_dir / "query.jpg", query_db.vis_sample(0))
    for topi, res in enumerate(query_results):
        idx = res.entry_idx
        dist = res.dist

        write_img(out_dir / f"top_{topi+1}.jpg", gallery.vis_sample(idx, label=str(dist)))


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
