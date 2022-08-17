
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
from pathlib import Path
from scripts.default_config import get_default_config

from utils.reid import PRWDataset, ReIdGallery, draw_box
from utils.reid_predictor import ReIdPredictor

# from yolov5.yolo_predictor import Yolov5Predictor

def load_cfg(config_file, weights):
    cfg = get_default_config()
    cfg.merge_from_file(config_file)
    cfg.use_gpu = False
    cfg.data.root = "../data/"
    cfg.model.load_weights = weights
    cfg.test.evaluate = True
    return cfg


def vis_query_res(data, query_img, query_res):
    keys, dists, boxes = zip(*[(res["key"], res["dist"], res["box"]) for res in query_res])
    # seq_name, keys = zip(*seq_keys)

    scale = 0.25
    gal_imgs = [data.get_frame_by_key(k)["image"] for k in keys]
    gal_imgs, boxes = zip(*[
        (cv2.resize(img, None, fx=scale, fy=scale), box * scale) 
    for img, box in zip(gal_imgs, boxes)])

    height = max(img.shape[0] for img in ([query_img] + list(gal_imgs)))

    gal_img_vis = [draw_box(img, box, f"{dist:.2f}") for img, dist, box in zip(gal_imgs, dists, boxes)]

    imgs = [np.pad(img, ((0, height - img.shape[0]), (0,10), (0,0))) for img in ([query_img] + gal_img_vis)]
    out = np.concatenate(imgs, 1)

    return out


def test_equality(args):
    import torch.nn.functional as F

    def model_build_func():
        cfg = load_cfg(args.cfg, args.weights)
        name = Path(args.cfg).stem
        return ReIdPredictor(name=name, cfg=cfg, skip_preproc=False, output_type="np").to(args.device)

    x = np.random.uniform(0, 1, (256, 128, 3)).astype(np.uint8)

    model1 = model_build_func()
    model2 = model_build_func()

    a, b = model1(x), model2(x)
    print(np.sum((a - b) ** 2))
    assert np.allclose(a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/im_r50_softmax_256x128_amsgrad.yaml")
    parser.add_argument("--weights", default="pretrained/resnet50_market_xent.pth.tar")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_det", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg, args.weights)
    name = Path(args.cfg).stem

    # test_equality(args)

    # reid_dev = args.device
    # det_

    # if args.device 

    data = PRWDataset("../data/PRW-v16.04.20")
    feat_extr = ReIdPredictor(name=name, cfg=cfg, skip_preproc=False, output_type="np").to(args.device)

    if args.use_det:
        det_cfg = "runs/train/bytetrack_mot20_5data/weights/best.pt"
        det_weights = "models/yolov5m_warboy.yaml"
        det_input_size = (512, 512)
        det_batch_size = 4
        # TODO: select different device for person det and reid
        # det = Yolov5Predictor(name=Path(det_cfg).stem, cfg=det_cfg, weights=det_weights, input_size=det_input_size, 
        #     batch_size=det_batch_size, pad_batch=True).to(args.device)

    gallery_name = f"prw_gal_{args.device}"

    gal = ReIdGallery(feat_extr)
    if not gal.load(gallery_name, nonexist_ok=True):
        ind = data.get_cam_seq_indices()
        for cam_idx, seq_idx in tqdm(ind, desc="Creating gallery"):
            gal.add_sequence(data.iter_frames(cam_idx, seq_idx))
        gal.save(gallery_name)

    query_img = data.get_query_image(0)["image"]

    query_res = gal.query([query_img])[0]

    vis = vis_query_res(data, query_img, query_res)
    cv2.imwrite("out.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
