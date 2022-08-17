import cv2
from pathlib import Path
import re
import torch
import numpy as np
import pickle
from scipy.io import loadmat
from ai_util.vision.image_gallery import ImageGallery
from ai_util.vision.image_gallery_reader import ImageGalleryReader


def batch_iter(it, batch_size):
    if isinstance(it, list):
        it = iter(it)

    has_items = True
    while has_items:
        batch = []

        try:
            for _ in range(batch_size):
                batch.append(next(it))
        except StopIteration:
            has_items = False
        
        if len(batch) > 0:
            yield batch


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def _vis_box(img, xyxy, scale=1, label=None, track_id=None):
    xyxy = xyxy * scale

    if track_id is None:
        color = [225, 255, 255]
    else:
        color = colors(track_id)

    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    cv2.rectangle(img, (x1,y1), (x2, y2), color, thickness=tl, lineType=cv2.LINE_AA)

    tf = max(tl - 1, 1)  # font thickness

    if label is not None:
        cv2.putText(img, str(label), (x1, y1 - 2), 0, tl / 3, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
    return img


class ReIdGalleryReader(ImageGalleryReader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def draw(self, img, entries, dists, scale):
        for i, entry in enumerate(entries):
            img = _vis_box(img, entry["box"], scale=scale, label=dists[i] if dists is not None else "", track_id=entry.get("track_id", None))

        return img


class ReIdGallery(ImageGallery):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def process_data_sample(self, sample):
        key = sample["key"]
        img = sample["image"]
        box = sample["box"]

        if sample["track_id"] is not None:
            box = np.concatenate([box, [sample["track_id"]]], 0)

        yield key, img, box

    def retrieve_data(self, idx, comp_aligned_img=True):
        data = super().retrieve_data(idx)
        if comp_aligned_img:
            x1, y1, x2, y2 = data["box"][:4]
            image_cropped = data["image"][y1:y2, x1:x2]
            data["image_cropped"] = image_cropped
        return data

    def visualize(self, idx):
        return self.vis_sample(idx)

    def vis_sample(self, idx, label=None):
        sample = self.retrieve_data(idx)

        img = sample["image"]
        box = sample["box"]

        img = _vis_box(img, box, label=label)

        return img

    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        key = d["key"]
        feat = d["feat"]
        box = d["box"][:4]

        if d["box"].shape[0] > 4:
            track_id = d["box"][4]
        else:
            track_id = None

        return {
            "key": key,
            "feat": feat,
            "box": box,
            "track_id": track_id
        }

    def create_reader(self, *args, **kwargs):
        return ReIdGalleryReader(self, *args, **kwargs)


class PRWDataset:
    def __init__(self, root_dir) -> None:
        self.root_dir = Path(root_dir)
        img_dir = self.root_dir / "frames"
        self.label_cache_file = self.root_dir / "label_cache.pkl"

        img_files = sorted(img_dir.glob("*.jpg"))

        self.cam_seq_img_files = [[]]

        reg = re.compile("c(\d+)s(\d+)_(\d+)")
        for img_file in img_files:
            res = reg.match(img_file.stem)
            cam_idx, seq_idx, frame_id = int(res.group(1)) - 1, int(res.group(2)) - 1, int(res.group(3))

            while len(self.cam_seq_img_files) <= cam_idx:
                self.cam_seq_img_files.append([])

            while len(self.cam_seq_img_files[cam_idx]) <= seq_idx:
                self.cam_seq_img_files[cam_idx].append([])

            self.cam_seq_img_files[cam_idx][seq_idx].append({
                "frame_id": frame_id,
                "img_file": img_file
            })

        if not self.label_cache_file.exists():
            cam_seq_annot = [[[
                self.read_annot(img_file["img_file"]) for img_file in img_files] 
                for img_files in seq_img_files] 
                for seq_img_files in self.cam_seq_img_files]

            with open(self.label_cache_file, "wb") as f:
                pickle.dump(cam_seq_annot, f)

        with open(self.label_cache_file, "rb") as f:
            self.cam_seq_annot = pickle.load(f)

        self.query_imgs = self._load_query_imgs()

    def _load_query_imgs(self):
        query_imgs = []

        reg = re.compile("(\d+)_c(\d+)s(\d+)_(\d+)")
        query_img_files = sorted((self.root_dir / "query_box").glob("*.jpg"))

        for img_file in query_img_files:
            res = reg.match(img_file.stem)
            track_id, cam_idx, seq_idx, frame_id = int(res.group(1)), int(res.group(2)) - 1, int(res.group(3)) - 1, int(res.group(4))
            query_imgs.append({
                "img_file": img_file,
                "track_id": track_id,
                "cam_idx": cam_idx,
                "seq_idx": seq_idx,
                "frame_id": frame_id,
            })

        return query_imgs

    def read_annot(self, img_path):
        annot_file = str(img_path.with_suffix(".jpg.mat"))
        annot_file = annot_file.replace("/frames/", "/annotations/")
        data = loadmat(annot_file)

        box_annot = None

        for key in ("box_new", "anno_file", "anno_previous"):
            if key in data:
                box_annot = data[key]
                break

        assert box_annot is not None and box_annot.ndim == 2 and box_annot.shape[1] == 5

        boxes_ids = box_annot[:, [1, 2, 3, 4, 0]].astype(np.float32)  # push id back

        # boxes_ids[:, 0:2] = boxes_ids[:, 0:2]
        boxes_ids[:, 2:4] = boxes_ids[:, 0:2] + boxes_ids[:, 2:4]

        return boxes_ids

    def _load_img(self, path):
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

    def get_frame_by_key(self, key):
        return self.get_frame(*key)

    def get_frame(self, cam_idx, seq_idx, frame_idx):
        d = self.cam_seq_img_files[cam_idx][seq_idx][frame_idx]
        img_file = d["img_file"]
        frame_id = d["frame_id"]
        img = self._load_img(img_file)
        box = self.cam_seq_annot[cam_idx][seq_idx][frame_idx]

        return {
            "key": (cam_idx, seq_idx, frame_idx),
            "image": img,
            "box": box
        }

    def get_query_image(self, idx):
        d = self.query_imgs[idx]
        img = self._load_img(d["img_file"])
        info_keys = ["track_id", "cam_idx", "seq_idx", "frame_id"]

        return {
            "image": img,
            **{k: d[k] for k in info_keys}
        }

    def get_cam_seq_indices(self):
        return [(c, s) for c in range(len(self.cam_seq_img_files)) for s in range(len(self.cam_seq_img_files[c]))]

    def iter_frames(self, cam_idx, seq_idx):
        for i in range(len(self.cam_seq_img_files[cam_idx][seq_idx])):
            yield self.get_frame(cam_idx, seq_idx, i)


def draw_boxes(img, boxes):
    color = (255, 0, 0)
    for x1, y1, x2, y2 in boxes[:, :4].astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=1)


def draw_box(img, box, label):
    tl = 3
    tf = 1

    color = (255, 0, 0)
    x1, y1, x2, y2 = box[:4].astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=1)
    cv2.putText(img,
                label, (10, 30),
                0,
                tl / 3, [225, 0, 0],
                thickness=tf,
                lineType=cv2.LINE_AA)

    return img
