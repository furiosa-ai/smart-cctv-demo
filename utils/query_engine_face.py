

import os
from ai_util.dataset import ImageDataset
from utils.query_engine_base import QueryEngineBase


# from insightface.face_recognition import ArcFacePredictor, FaceExtractor, FaceGallery

# from yolov5_face.face_predictor import Yolov5FacePredictor

from ext_modules import ArcFacePredictor, FaceExtractor, FaceGallery, Yolov5FacePredictor

class QueryEngineFace(QueryEngineBase):
    def __init__(self, device="cpu", calib_mode=None, topk=5) -> None:
        super().__init__(topk=topk)

        if device == "furiosa":
            det_dev = "furiosa:npu0pe0"
            reg_dev = "furiosa:npu0pe1"
        else:
            det_dev, reg_dev = device, device

        self.face_det = Yolov5FacePredictor(cfg="yolov5_face/models/yolov5m_relu.yaml", weights="yolov5_face/weights/my/yolov5_relu.pt", 
            input_format="chw", input_prec="f32", calib_mode=calib_mode, quant_tag=calib_mode).to(det_dev)

        self.feat_extr = ArcFacePredictor(cfg="configs/ms1mv3_r50_leakyrelu.py", weights="insightface/runs/ms1mv3_r50_leakyrelu_1/model.pt", 
            input_format="chw", input_prec="f32", calib_mode=calib_mode, quant_tag=calib_mode,
            normalize=True, batch_size=1, pad_batch=True).to(reg_dev)

        self.gallery = None
        self.path = None

    def set_gallery_data(self, path):
        self.path = path
        self.gallery_data = ImageDataset(path, limit=None, frame_step=1)
        self.process_gallery_data()
        return self.gallery_data

    def process_gallery_data(self):
        gallery_cache_name = os.path.abspath(self.path).replace("/", "_")
        print(gallery_cache_name)
        self.gallery = FaceGallery(
            name=gallery_cache_name, 
            data=self.gallery_data, 
            data_extr=FaceExtractor(self.face_det), 
            feat_extr=self.feat_extr,
            gallery_dir="galleries/face/"
        )

    def create_query_db_from_image_files(self, img_files):        
        query_db = FaceGallery(
            data=ImageDataset(img_files), 
            data_extr=FaceExtractor(self.face_det, single_face=True), 
            feat_extr=self.feat_extr
        )

        return query_db

    def query(self, query):
        query_results, distmat = self.gallery.query([query["feat"]], topk=self.topk, return_distmat=True)
        query_results = query_results[0]
        distmat = distmat[:1]
        return query_results, distmat
