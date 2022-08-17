
from math import ceil
import ai_util.inference_framework
import ai_util.data_sampler
from ai_util.imp_env import ImpEnv
from ai_util.dataset import ImageDataset
import argparse
import time
import numpy as np
import torch
import easydict
import os
import sys
import cv2
from pathlib import Path
import mxnet
import pandas
import multiprocessing

# sys.path.insert(0, "yolov5_face")
# from yolov5_face.face_predictor.face_predictor import Yolov5FacePredictor


with ImpEnv("yolov5_face", imports_keep=["models", "models.yolo", "yolov5_face.face_predictor.face_predictor"]):
    from yolov5_face.face_predictor.face_predictor import Yolov5FacePredictor

with ImpEnv(imports_keep=["recognition.arcface_torch.utils.face_extract", "recognition.arcface_torch.utils.arcface_predictor"]):
    from recognition.arcface_torch.utils.arcface_predictor import ArcFacePredictor
    from recognition.arcface_torch.utils.face_extract import FaceExtractor

with ImpEnv():
    from recognition.arcface_torch.utils.face_gallery import FaceGallery


def show_img(img):
    cv2.imshow("out", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def read_img(img_file):
    return cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB)


def write_img(img_file, img):
    return cv2.imwrite(str(img_file), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def build_gallery_part(idx, data_path, face_det, feat_extr, start_idx=0, end_idx=None):
    face_det = face_det.shared_inference(idx=idx)
    feat_extr = feat_extr.shared_inference(idx=idx)

    data = ImageDataset(data_path, frame_step=5, start_idx=start_idx, end_idx=end_idx)

    gal = FaceGallery(
        name=None,  # dont save path
        data=data, 
        data_extr=FaceExtractor(face_det), 
        feat_extr=feat_extr
    )

    return gal


class ProcessPool:
    def __init__(self, target, args) -> None:
        self.target = target
        self.num_proc = len(args)

        self.qu = multiprocessing.Queue(maxsize=self.num_proc)

        self.procs = [multiprocessing.Process(target=self._task, args=(i, self.qu, *a)) for i, a in enumerate(args)]

    def _task(self, idx, out_qu, *args):
        res = self.target(*args)
        # out_qu.put((idx, res))
        out_qu.put((idx, res))
        print("send")

    def start(self):
        for proc in self.procs:
            proc.start()

    def join(self):
        for _ in range(self.num_proc):
            idx, res = self.qu.get()
            print("recv")
            out[idx] = res

        for proc in self.procs:
            proc.join()

        out = [None] * len(self.procs)


        return out


def build_gallery(name, data_path, face_det, feat_extr, num_worker):
    data = ImageDataset(data_path, frame_step=5)

    gallery = FaceGallery(
        name=name,
        data=data, 
        data_extr=None, 
        feat_extr=feat_extr
    )

    if gallery.is_loaded:
        return gallery
    else:
        data_len = len(data)
        data_per_worker = ceil(data_len / max(num_worker, 1))
        data_len = 20
        start_end_ind = [(i * data_per_worker, min((i + 1) * data_per_worker, data_len)) for i in range(max(num_worker, 1))]
        proc_args = [(i, data_path, face_det, feat_extr, s, e) for i, (s, e) in enumerate(start_end_ind)]

        proc_group = ProcessPool(target=build_gallery_part, args=proc_args)

        proc_group.start()

        face_det.start_inference()
        feat_extr.start_inference()

        galleries_part = proc_group.join()

        face_det.end_inference()
        feat_extr.end_inference()

        #with multiprocessing.Pool(processes=num_worker) as pool:
        #     galleries_part = pool.starmap(build_gallery_part, [(data_path, face_det, feat_extr, s, e) for (s, e) in start_end_ind])

        gallery.add_data_from_galleries(galleries_part)

    return gallery


class PredictorBuilder:
    def __init__(self, face_det_calib, feat_extr_calib):
        self.face_det_calib = face_det_calib
        self.feat_extr_calib = feat_extr_calib

    def __call__(self):
        face_det = Yolov5FacePredictor(cfg="yolov5_face/models/yolov5m_relu.yaml", weights="yolov5_face/weights/my/yolov5_relu.pt", 
            input_format="chw", input_prec="f32", calib_mode=self.face_det_calib, quant_tag=self.face_det_calib)

        feat_extr = ArcFacePredictor(cfg="configs/ms1mv3_r50_leakyrelu.py", weights="runs/ms1mv3_r50_leakyrelu_1/model.pt", 
            input_format="chw", input_prec="f32", calib_mode=self.feat_extr_calib, quant_tag=self.feat_extr_calib,
            normalize=True, batch_size=1, pad_batch=True)

        return face_det, feat_extr


def query_demo(args):
    manager = multiprocessing.Manager()

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

    num_worker = 1

    
    face_det = Yolov5FacePredictor(cfg="yolov5_face/models/yolov5m_relu.yaml", weights="yolov5_face/weights/my/yolov5_relu.pt", 
        input_format="chw", input_prec="f32", calib_mode=face_det_calib, quant_tag=face_det_calib).shared_inference(device=det_dev, num_inst=num_worker)

    feat_extr = ArcFacePredictor(cfg="configs/ms1mv3_r50_leakyrelu.py", weights="runs/ms1mv3_r50_leakyrelu_1/model.pt", 
        input_format="chw", input_prec="f32", calib_mode=feat_extr_calib, quant_tag=feat_extr_calib,
        normalize=True, batch_size=1, pad_batch=True).shared_inference(device=reg_dev, num_inst=num_worker)

    gallery_cache_name = os.path.abspath(args.gallery).replace("/", "_") + (f"_{args.calib_mode}" if args.calib_mode is not None else "")
    gallery = build_gallery(
        name=gallery_cache_name, 
        data_path=args.gallery, 
        face_det=face_det, 
        feat_extr=feat_extr,
        num_worker=num_worker
    )
    
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
    # multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--calib_mode")
    args = parser.parse_args()

    query_demo(args)


if __name__ == "__main__":
    main()
