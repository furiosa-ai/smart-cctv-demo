from math import ceil
import numpy as np
import skimage.transform
import cv2
from tqdm import tqdm
from ai_util.data_loader import DataLoader
from ai_util.data_sampler import DataSampler, VideoDataSampler
from ai_util.dataset import ImageDataset, batch_iter
from multiprocessing import Process, Queue


class FaceAlign:
    def __init__(self) -> None:
        self.src_lm = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
            ], dtype=np.float32)
        self.src_lm[:, 0] += 8.0

    def __call__(self, img, landmark):
        facial5points = np.array(landmark, dtype=np.float32).reshape(5, 2)

        st = skimage.transform.SimilarityTransform()
        st.estimate(facial5points, self.src_lm)
        img = cv2.warpAffine(img, st.params[0:2, :], (112, 112), borderValue=0.0)
        
        return img


def extract_faces(data_loader, face_det, single_face=False):
    face_align = FaceAlign()

    if isinstance(data_loader, DataLoader):
        dataset = data_loader.data
        data_size = dataset.get_key_frame_count()
        dataset = iter(dataset)
        num_batches = ceil(data_size / face_det.batch_size)
    else:
        num_batches = None

    def _input_iter():
        for batch in data_loader:
            keys, imgs = zip(*batch)
            yield imgs, (keys, imgs)

    for preds, (keys, imgs) in face_det.predict_all(_input_iter()):
        for key, img, pred in zip(keys, imgs, preds):
            if single_face:
                if len(pred) == 0:
                    yield None
                else:
                    pred = pred[:1]

            for box, conf, lm in pred:
                box = box[:4].astype(int)
                img_aligned = face_align(img, lm)
                yield {
                    "key": key,
                    "image_full": img,
                    "image": img_aligned,
                    "box": box,
                    "lm": lm
                }





def create_subset_sampler(data_sampler, subset_idx, num_subsets):
    class SubsetSampler(DataSampler):
        def __init__(self, data, worker_idx, num_workers) -> None:
            assert worker_idx == 0
            assert num_workers == 0
            super().__init__(data, worker_idx, num_workers)

            self.sampler = data_sampler(data, subset_idx, num_subsets)
            self.it = None

        def __iter__(self):
            self.it = iter(self.sampler)
            return self

        def __next__(self):
            return next(self.it)
            
    return SubsetSampler


class FaceExtractor:
    def __init__(self, face_det, single_face=False, num_worker=2, data_sampler=VideoDataSampler) -> None:
        self.face_det = face_det
        self.single_face = single_face
        self.num_worker = num_worker
        self.data_sampler = data_sampler

    def __call__(self, data):
        data_loader = DataLoader(data, self.face_det.batch_size, num_worker=0, data_sampler=self.data_sampler)
        face_gen = extract_faces(data_loader, self.face_det, self.single_face)
        return face_gen


class FaceExtractorParallel:
    def __init__(self, face_det, single_face=False, num_worker=2, data_sampler=VideoDataSampler) -> None:
        self.face_det = face_det
        self.single_face = single_face
        self.num_worker = num_worker
        self.data_sampler = data_sampler

    def data_worker(self, idx, qu: Queue):
        data_loader = DataLoader(self.data, self.face_det.batch_size, num_worker=0, 
            data_sampler=create_subset_sampler(self.data_sampler, idx, self.num_worker))
        # data = self.data if idx == 0 else copy.deepcopy(self.data)
        # from ai_util.dataset import ImageDataset
        # data = ImageDataset("../data/test_face/tc1/tom_cruise_test.mp4", limit=None, frame_step=5)

        for sample in data_loader:
            assert sample is not None
            qu.put(sample)

        qu.put(None)

    def __call__(self, data):
        """
        data_loaders = [DataLoader(data, self.face_det.batch_size, num_worker=0, 
            data_sampler=create_subset_sampler(self.data_sampler, i, self.num_worker)) 
            for i in range(self.num_worker)
        ]
        """

        data_loader = DataLoader(data, self.face_det.batch_size, num_worker=0, data_sampler=self.data_sampler)
        face_gen = extract_faces(data_loader, self.face_det, self.single_face)
        return face_gen
