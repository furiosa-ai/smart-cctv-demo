
import numpy as np
import cv2
from pathlib import Path

from tqdm import tqdm
from ai_util.vision.image_gallery import ImageGallery
from ai_util.vision.image_gallery_reader import ImageGalleryReader

from bin_to_np import load_np
from recognition.arcface_torch.utils.face_extract import FaceAlign




def read_img(img_file):
    return cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB)


def _vis_box_lm(img, xyxy, landmarks, scale=1, label=None):
    xyxy = xyxy * scale
    landmarks = landmarks * scale

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

    if label is not None:
        cv2.putText(img, str(label), (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


class FaceGalleryReader(ImageGalleryReader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def draw(self, img, entries, dists, scale):
        for i, entry in enumerate(entries):
            img = _vis_box_lm(img, entry["box"], entry["lm"], scale=scale, label=dists[i] if dists is not None else "")

        return img
    

class FaceGallery(ImageGallery):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(dist_metric="cosine", *args, **kwargs)
        self.face_algin = FaceAlign()

    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        key = d["key"]
        feat = d["feat"]
        box = d["box"][:4]
        lm = d["box"][4:]

        return {
            "key": key,
            "feat": feat,
            "box": box,
            "lm": lm
        }

    def retrieve_data(self, idx, comp_aligned_img=True):
        data = super().retrieve_data(idx)
        if comp_aligned_img:
            img_aligned = self.face_algin(data["image"], data["lm"])
            data["image_aligned"] = img_aligned
        return data

    def process_data_sample(self, sample):
        key = sample["key"]
        img = sample["image"]
        box_lm = np.concatenate([sample["box"], sample["lm"]])

        yield key, img, box_lm

    def vis_box_lm(self, idx):
        sample = self.retrieve_data(idx)
        img = sample["image"].copy()
        img = _vis_box_lm(img, sample["box"], sample["lm"])
        return img
    
    def visualize(self, idx):
        return self.vis_box_lm(idx)

    def vis_sample(self, idx, label=None):
        sample = self.retrieve_data(idx)

        img = sample["image"]
        img_aligned = sample["image_aligned"]

        if img.shape[0] > 480:
            s = 480 / img.shape[0]
            img = cv2.resize(img, None, fx=s, fy=s)
        else:
            s = 1

        img = _vis_box_lm(img, sample["box"], sample["lm"], scale=s, label=label)

        h = max(img.shape[0], img_aligned.shape[0])
        img = np.pad(img, ((0, h - img.shape[0]), (0, 0), (0, 0)))
        img_aligned = np.pad(img_aligned, ((0, h - img_aligned.shape[0]), (0, 0), (0, 0)))

        out = np.concatenate([img, img_aligned], 1)

        return out

    def create_reader(self, *args, **kwargs):
        return FaceGalleryReader(self, *args, **kwargs)


class FaceGalleryOld:
    def __init__(self, emb_size=512) -> None:
        self.emb_size = emb_size
        self.emb_mat = np.zeros((0, emb_size), dtype=np.float32)
        self.imgs = []
        self.names = []

    def __len__(self):
        return self.emb_mat.shape[0]

    def add(self, mat, imgs, names):
        self.emb_mat = np.concatenate((self.emb_mat, mat), 0)
        self.imgs += imgs
        self.names += names

    def add_from_np(self, path):
        imgs = load_np(path, (112, 112))
        pass

    def add_from_folder(self, face_rec_pred, path, limit=None):
        dirs = sorted(Path(path).glob("*/"))

        if limit is not None:
            dirs = dirs[:limit]

        names = []
        feats = []
        imgs = []

        for d in tqdm(dirs, desc="Adding to database"):
            name = d.stem
            img_file = sorted(d.glob("*"))[0]
            img = read_img(img_file)
            feat = face_rec_pred(img)

            names.append(name)
            imgs.append(img)
            feats.append(feat)

        feats = np.stack(feats)

        self.add(feats, imgs, names)

    def add_from_image(self):
        pass

    def get_name(self, idx):
        return self.names[idx]

    def get_image(self, idx):
        return self.imgs[idx]

    def query(self, emb, top_n):
        emb = emb / np.linalg.norm(emb)
        cos_sim = np.dot(self.emb_mat, emb)
        ind = (-cos_sim).argsort()[:top_n]
        ind_sims = cos_sim[ind]
        # ind_sims = np.arccos(cos_sim[ind])
        return ind, ind_sims