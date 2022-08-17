import numpy as np
import skimage.transform
import cv2
from ai_util.dataset import ImageDataset, batch_iter
from tqdm import tqdm
from math import ceil


def extract_boxes(dataset, box_det, single_box=False, min_box_size=10, tracker=None):
    if isinstance(dataset, ImageDataset):
        data_size = dataset.get_key_frame_count()
        dataset = iter(dataset)
        num_batches = ceil(data_size / box_det.batch_size)
    else:
        num_batches = None

    def _input_iter():
        for batch in batch_iter(dataset, box_det.batch_size):
            keys, imgs = zip(*batch)
            yield imgs, (keys, imgs)

    for preds, (keys, imgs) in box_det.predict_all(_input_iter()):
        for key, img, pred in zip(keys, imgs, preds):
            if single_box:
                if len(pred) == 0:
                    yield None
                else:
                    pred = pred[:1]

            if tracker is not None:
                pred = tracker(pred)

            for p in pred:
                if tracker is not None:
                    box, conf, cls, track_id = p[:4].astype(int), None, None, int(p[4])
                else:
                    box, conf, cls, track_id = p[:4].astype(int), p[4], int(p[5]), None
                x1, y1, x2, y2 = box
                img_cropped = img[y1:y2, x1:x2]

                if img_cropped.shape[0] >= min_box_size and img_cropped.shape[1] >= min_box_size:
                    yield {
                        "key": key,
                        "image_full": img,
                        "image": img_cropped,
                        "box": box,
                        "conf": conf,
                        "class": cls,
                        "track_id": track_id,
                    }
    

class BoxExtractor:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data):
        face_gen = extract_boxes(iter(data), *self.args, **self.kwargs)
        return face_gen 


class BoxExtractorIdentity:
    def __init__(self) -> None:
        pass

    def __call__(self, data):
        def _extract():
            for key, img in data:
                yield {
                    "key": key,
                    "image_full": img,
                    "image": img,
                    "box": np.array([0, 0, img.shape[1], img.shape[0]], dtype=int),
                    "conf": 1.0,
                    "class": None,
                    "track_id": None,
                }

        return _extract()