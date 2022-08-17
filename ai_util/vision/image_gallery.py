from collections import OrderedDict, namedtuple
import cv2
from pathlib import Path
import re
import torch
import numpy as np
import pickle
from scipy.io import loadmat
from ai_util.dataset import batch_iter
from tqdm import tqdm

from torch.nn import functional as F

from ai_util.vision.image_gallery_reader import ImageGalleryReader



def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1 = torch.from_numpy(input1)
    input2 = torch.from_numpy(input2)

    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)

    distmat = distmat.numpy()

    return distmat


def cosine_distance(input1, input2):
    # input1 = input1 / np.linalg.norm(input1, axis=1, keepdims=True)
    # input2 = input2 / np.linalg.norm(input2, axis=1, keepdims=True)
    prod = np.dot(input1, input2.T)
    return 1 - prod


QueryResult = namedtuple("QueryResult", ("entry_idx", "data_key", "dist"))


class ImageGallery:
    def __init__(self, data, data_extr=None, feat_extr=None, name=None, dist_metric=None, gallery_dir="galleries") -> None:
        self.name = name
        self.data = data
        self.key_idx_map = None  # one key has idx per box
        self.keys = []
        self.feats = None
        self.boxes = []  # need to store boxes too, and asign a crop it to it
        self.feat_extr = feat_extr
        self.min_box_size = 5
        self.dist_metric = dist_metric if dist_metric is not None else (feat_extr.dist_metric if feat_extr is not None else None)
        self.is_loaded = False
        self.key_freq = np.inf
        self.gallery_dir = Path(gallery_dir)

        self.gallery_dir.mkdir(parents=True, exist_ok=True)

        if name is None or not self.load(nonexist_ok=True):
            if data_extr is not None:
                self._extract_data(data_extr)
                if name is not None:
                    self.save()

    def _build_key_idx_map(self):
        self.key_idx_map = OrderedDict()

        for idx, key in enumerate(self.keys):
            if key not in self.key_idx_map:
                self.key_idx_map[key] = []

            self.key_idx_map[key].append(idx)

        last_key = -np.inf
        for key in self.key_idx_map:
            self.key_freq = min(self.key_freq, key - last_key)
            last_key = key

    def save(self):
        path = self.gallery_dir / f"{self.name}.npz"

        out = {
            "keys": np.array(self.keys),
            "feats": self.feats,
            "boxes": np.array(self.boxes)
        }

        np.savez(path, **out)

    def load(self, nonexist_ok=False):
        path = self.gallery_dir / f"{self.name}.npz"

        if not path.is_file():
            if nonexist_ok:
                self.is_loaded = False
            else:
                raise FileNotFoundError
        else:
            data = np.load(path)

            self.keys = [*data["keys"]]
            self.feats = data["feats"]
            self.boxes = [*data["boxes"]]

            self.is_loaded = True

        self._build_key_idx_map()
        return self.is_loaded

    def process_data_sample(self, sample):
        raise NotImplementedError

    def _iter_seq(self, seq):
        for sample in seq:
            for key, crop, box in self.process_data_sample(sample):
                yield key, crop, box

    def compute_distance_matrix(self, feats1, feats2):
        if self.dist_metric == 'euclidean':
            distmat = euclidean_squared_distance(feats1, feats2)
        elif self.dist_metric == 'cosine':
            distmat = cosine_distance(feats1, feats2)

        return distmat

    def __getitem__(self, idx):
        return {
            "key": self.keys[idx],
            "box": self.boxes[idx],
            "feat": self.feats[idx]
        }

    def retrieve_data(self, idx, **kwargs):
        d = self.__getitem__(idx)
        img = self.data.load_image_by_key(d["key"])

        return {
            **d,
            "image": img
        }

    def query(self, query_feats, topk=5, return_distmat=False):
        """
        data = batch_iter(query_data, self.feat_extr.batch_size)
        query_feats = []
        for batch in data:
            out = self.feat_extr(batch)
            query_feats += [*out]
        """
        query_feats = np.stack(query_feats)

        distmat = self.compute_distance_matrix(query_feats, self.feats)
        indices = np.argsort(distmat, axis=1)
        indices_topk = indices[:, :topk]

        """
        keys = [[self.keys[i] for i in row] for row in indices_topk]
        boxes = [[self.boxes[i] for i in row] for row in indices_topk]
        dists = [distmat[i, idx] for i, idx in enumerate(indices_topk)]
        """

        """
        out = [[{
                **self.__getitem__(i),
                "dist": distmat[row_idx, i],
            }
            for i in row] for row_idx, row in enumerate(indices_topk)]
        """

        out = [[QueryResult(
                entry_idx=i,
                data_key=self.keys[i],
                dist=distmat[row_idx, i],
            )
            for i in row] for row_idx, row in enumerate(indices_topk)]

        if return_distmat:
            return out, distmat
        else:
            return out

    def add_data_from_galleries(self, gals):
        for gal in gals:
            self.keys += gal.key
            self.boxes += gal.boxes

            self.feats = np.concatenate(
                [self.feats, gal.feats] if self.feats is not None else gal.feats,
                0
            )

        self._build_key_idx_map()

    def _extract_data(self, data_iterator):
        seq = data_iterator(self.data)
        data = batch_iter(self._iter_seq(seq), self.feat_extr.batch_size)

        new_keys = []
        new_boxes = []
        new_feats = []

        """
        for batch in data:
            keys, crops, boxes = zip(*batch)
            feats = self.feat_extr(crops)
            new_keys += keys
            new_boxes += boxes
            new_feats += [f[None] for f in feats]  # unwrap batch
        """

        def _feat_input_iter():
            for batch in data:
                keys, crops, boxes = zip(*batch)
                yield crops, (keys, boxes)

        for feats, (keys, boxes) in tqdm(self.feat_extr.predict_all(_feat_input_iter()), desc="Creating gallery"):
            new_keys += keys
            new_boxes += boxes
            new_feats += [f[None] for f in feats]  # unwrap batch

        self.keys += new_keys
        self.boxes += new_boxes

        self.feats = np.concatenate(
            [self.feats, *new_feats] if self.feats is not None else new_feats,
            0
        )

        self._build_key_idx_map()

        return self

    def create_reader(self, *args, **kwargs):
        return ImageGalleryReader(self, *args, **kwargs)


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
