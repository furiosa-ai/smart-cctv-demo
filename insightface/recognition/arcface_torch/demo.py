import argparse
import itertools
import numpy as np
import torch
import cv2
from pathlib import Path
from torch.autograd import Variable
import skimage.transform

from tqdm import tqdm

from backbones import get_model
from recognition.arcface_torch.test import ArcFacePredictor
from recognition.arcface_torch.utils.face_dataset import FaceDataset, FaceExtractor, extract_faces
from recognition.arcface_torch.utils.face_gallery import FaceGallery
from recognition.arcface_torch.utils.face_reg_pipeline import FaceRecognitionPipeline
from recognition.arcface_torch.utils.mtcnn_face_det import MTCNNFaceDetector
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from bin_to_np import load_np

from PIL import Image
from mtcnn.get_nets import PNet, RNet, ONet
from mtcnn.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from mtcnn.first_stage import run_first_stage

from yolov5_face.face_predictor import Yolov5FacePredictor


def show_img(img):
    cv2.imshow("out", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def read_img(img_file):
    return cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB)


def write_img(img_file, img):
    return cv2.imwrite(str(img_file), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


"""
class ArcFacePredictor:
    def __init__(self, cfg, weights, device="cuda") -> None:
        if isinstance(cfg, str):
            cfg = get_config(cfg, [])
            # global control random seed
            setup_seed(seed=cfg.seed, cuda_deterministic=False)

        backbone = get_model(
            cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size, act=cfg.act).cuda().eval()

        checkpoint = torch.load(weights)

        backbone.load_state_dict(checkpoint)
        print(f"Loaded checkpoint {weights}")

        backbone = backbone.to(device)
        backbone.eval()
        self.model = backbone
        self.device = device

    def preproc(self, img):
        img_1 = np.expand_dims(img, 0)
        img_2 = np.expand_dims(np.fliplr(img), 0)
        output = np.concatenate((img_1, img_2), axis=0).astype(np.float32)
        output = np.transpose(output, (0, 3, 1, 2))
        output = ((output / 255) - 0.5) / 0.5
        output = torch.from_numpy(output).to(self.device)
        return output

    def __call__(self, img):
        with torch.no_grad():
            inputs = self.preproc(img)
            feat = self.model(inputs)
            feat = feat.cpu().numpy()
            
            # feat = np.reshape(feat, (-1, 512 * 2))[0]
            feat = np.sum([f / np.linalg.norm(f) for f in feat], 0)
            feat = feat / np.linalg.norm(feat)
            return feat
"""


def img_gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def vis_conf_matrix(rec_pipe, imgs_a, imgs_b, warp_a=True, warp_b=True):
    patch_shape = imgs_a[0].shape
    patches = []

    feats_a = np.stack([rec_pipe(img, warp_a)["feat"] for img in imgs_a], 0)
    feats_b = np.stack([rec_pipe(img, warp_b)["feat"] for img in imgs_b], 0)
    
    sim = np.dot(feats_a, feats_b.T)

    for y in range(-1, len(imgs_a)):
        for x in range(-1, len(imgs_b)):
            patch = None

            if y == -1 and x == -1:
                patch = np.zeros(patch_shape, dtype=np.uint8)
            elif y == -1:
                patch = imgs_b[x]
            elif x == -1:
                patch = imgs_a[y]
            else:
                s = sim[y][x]
                patch = np.full(patch_shape, round((s * 0.5 + 0.5) * 255), dtype=np.uint8)
                cv2.putText(patch, f"{s:.2f}", (patch_shape[1] // 4, patch_shape[0] // 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 1)
    
            patches.append(patch)

    grid = np.stack(patches)
    grid = img_gallery(grid, ncols=len(imgs_b) + 1)

    return grid


def read_first_img(path, limit):
    dirs = sorted(Path(path).glob("*/"))

    if limit is not None:
        dirs = dirs[:limit]

    imgs = []

    for d in tqdm(dirs, desc="Adding to database"):
        # name = d.stem
        img_file = sorted(d.glob("*"))[0]
        img = read_img(img_file)

        imgs.append(img)

    return imgs


def make_low_res_img(img, size):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
    return img


def make_low_res_imgs(img, sizes):
    return [make_low_res_img(img, s) for s in sizes]


class RMFDDataset:
    def __init__(self, root_dir, transform=None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.ids = [p.stem for p in (self.root_dir / "AFDB_face_dataset").glob("*")]

    def get_images(self, idx, count=None, masked=False):
        name = self.ids[idx]

        path = self.root_dir / ("AFDB_face_dataset" if not masked else "AFDB_masked_face_dataset") / name
        img_files = sorted(path.glob("*"))

        imgs = (read_img(i) for i in img_files)

        if self.transform is not None:
            imgs = (self.transform(i) for i in imgs)
            imgs = (i for i in imgs if i is not None)

        imgs = list(itertools.islice(imgs, count) if count is not None else imgs)

        return imgs


def demo1(args):
    if args.device == "furiosa":
        det_dev = "furiosa:npu0pe0"
        reg_dev = "furiosa:npu0pe1"
    else:
        det_dev, reg_dev = args.device, args.device

    # cfg = "configs/ms1mv3_r100.py"
    # weights = "weights/arcface/ms1mv3_arcface_r100_fp16.pth"
    cfg = args.config
    weights = args.weights

    # face_det = MTCNNFaceDetector()
    face_det = Yolov5FacePredictor("yolov5_face/weights/yolov5m-face.pt", input_format="chw", input_prec="f32")
    face_det.to(det_dev)

    face_rec_pred = ArcFacePredictor(cfg=cfg, weights=weights, name=Path(cfg).stem, normalize=True, preproc_input=True)
    face_rec_pred.to(reg_dev)
    # face_rec_pred.to("furiosa")

    database = FaceGallery()
    # database.add_from_folder(face_rec_pred, "../data/MS-Celeb-1M_Align_112x112/imgs", 100)

    rec_pipe = FaceRecognitionPipeline(
        face_det=face_det,
        face_rec_pred=face_rec_pred,
        database=database
    )

    # img = read_img("../data/MS-Celeb-1M_Align_112x112/imgs/43073/3032097.jpg")
    # img = read_img("../data/MS-Celeb-1M_Align_112x112/imgs/0/32.jpg")
    # img = read_img("../data/MS-Celeb-1M_Align_112x112/imgs/0/1.jpg")
    # img = read_img("../data/MS-Celeb-1M_Align_112x112/imgs/10040/782495.jpg")
    # out = rec_pipe(img)

    img_set1 = read_first_img("../data/MS-Celeb-1M_Align_112x112/imgs", 10)
    img_set2 = [read_img(f) for f in sorted(Path("../data/MS-Celeb-1M_Align_112x112/imgs/0/").glob("*"))[:10]]
    img_set3 = [make_low_res_img(img_set2[0], s) for s in (112, 56, 28, 19, 14)]

    grid1 = vis_conf_matrix(rec_pipe, img_set1, img_set1, False, False)
    grid2 = vis_conf_matrix(rec_pipe, img_set2, img_set2, False, False)
    grid3 = vis_conf_matrix(rec_pipe, img_set3, img_set2, False, False)

    write_img("out/diff_person.png", grid1)
    write_img("out/same_person.png", grid2)
    write_img("out/low_res.png", grid3)


def demo2(args):
    if args.device == "furiosa":
        det_dev = "furiosa:npu0pe0"
        reg_dev = "furiosa:npu0pe1"
    else:
        det_dev, reg_dev = args.device, args.device

    # cfg = "configs/ms1mv3_r100.py"
    # weights = "weights/arcface/ms1mv3_arcface_r100_fp16.pth"
    cfg = args.config
    weights = args.weights

    # face_det = MTCNNFaceDetector()
    face_det = Yolov5FacePredictor("yolov5_face/weights/my/yolov5_relu.pt", "yolov5_face/models/yolov5m_relu.yaml", input_format="chw", input_prec="f32")
    face_det.to(det_dev)

    feat_extr = ArcFacePredictor(cfg=cfg, weights=weights, name=Path(cfg).stem, normalize=True, preproc_input=True)
    feat_extr.to(reg_dev)
    # face_rec_pred.to("furiosa")

    data = FaceDataset("../data/WIDERFACE/val/*/*", limit=10)
    gallery = FaceGallery(name="test", data=data, feat_extr=feat_extr)
    # database.add_from_folder(face_rec_pred, "../data/MS-Celeb-1M_Align_112x112/imgs", 100)

    if not gallery.is_loaded:
        gallery.add_sequence(lambda d: extract_faces(iter(d), face_det))
        gallery.save()

    # sample = gallery.retrieve_data(0)

    write_img("out.jpg", gallery.vis_sample(0))

    """
    rec_pipe = FaceRecognitionPipeline(
        face_det=face_det,
        face_rec_pred=face_rec_pred,
        database=database
    )
    """


def demo3(args):
    if args.device == "furiosa":
        det_dev = "furiosa:npu0pe0"
        reg_dev = "furiosa:npu0pe1"
    else:
        det_dev, reg_dev = args.device, args.device

    # cfg = "configs/ms1mv3_r100.py"
    # weights = "weights/arcface/ms1mv3_arcface_r100_fp16.pth"
    cfg = args.config
    weights = args.weights

    # face_det = MTCNNFaceDetector()
    face_det = Yolov5FacePredictor("yolov5_face/weights/my/yolov5_relu.pt", "yolov5_face/models/yolov5m_relu.yaml", input_format="chw", input_prec="f32")
    face_det.to(det_dev)

    feat_extr = ArcFacePredictor(cfg=cfg, weights=weights, name=Path(cfg).stem, normalize=True, preproc_input=True, batch_size=1, pad_batch=True)
    feat_extr.to(reg_dev)
    # face_rec_pred.to("furiosa")

    gallery = FaceGallery(
        name="tom_cruise", 
        data=FaceDataset("../data/test_face/tc1/tom_cruise_test.mp4", limit=None, frame_step=5), 
        data_extr=FaceExtractor(face_det), 
        feat_extr=feat_extr
    )
    
    query_db = FaceGallery(
        data=FaceDataset(["../data/test_face/tc1/tom_cruise_profile1.jpeg"]), 
        data_extr=FaceExtractor(face_det, single_face=True), 
        feat_extr=feat_extr
    )

    query_results = gallery.query([query_db[0]["feat"]])[0]

    for topi, res in enumerate(query_results):
        # match_data = gallery.retrieve_data(res["idx"])
        idx = res["idx"]
        dist = res["dist"]

        write_img(f"top_{topi}.jpg", gallery.vis_sample(idx, label=str(dist)))


    # write_img("out.jpg", gallery.vis_sample(0))
    # write_img("out2.jpg", query_db.vis_sample(0))


def main():
    parser = argparse.ArgumentParser(
    description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    # demo1(args)
    demo3(args)


if __name__ == "__main__":
    main()
