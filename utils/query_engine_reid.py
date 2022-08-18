

import os
from ai_util.dataset import ImageDataset
from utils.query_engine_base import QueryEngineBase


# from torchreid.reid import ReIdPredictor, BoxExtractor, BoxExtractorIdentity, ReIdGallery

# from yolov5.yolo_predictor import Yolov5Predictor, Tracker

from ext_modules import ReIdPredictor, BoxExtractor, BoxExtractorIdentity, ReIdGallery, Yolov5Predictor, Tracker

class QueryEngineReId(QueryEngineBase):
    def __init__(self, device="cpu", topk=5, gallery_cache_builder=None) -> None:
        super().__init__(topk=topk)

        person_det_calib, feat_extr_calib = "entropy", "minmax"

        if device == "furiosa":
            det_dev = "furiosa:npu0pe0"
            reg_dev = "furiosa:npu0pe1"
        else:
            det_dev, reg_dev = device, device

        self.person_det = Yolov5Predictor(cfg="yolov5/models/yolov5m_warboy.yaml", weights="yolov5/runs/train/bytetrack_mot20_5data/weights/best.pt", 
            input_format="chw", input_prec="f32", calib_mode=person_det_calib, quant_tag=person_det_calib,
            input_size=(640, 640)).to(det_dev)

        self.feat_extr = ReIdPredictor(cfg="torchreid/configs/im_r50_softmax_256x128_amsgrad.yaml", weights="torchreid/pretrained/resnet50_market_xent.pth.tar",
            output_type="np", batch_size=1, pad_batch=True, calib_mode=feat_extr_calib, quant_tag=feat_extr_calib,).to(reg_dev)

        self.gallery = None
        self.path = None
        self.gallery_cache_builder = gallery_cache_builder

    def set_gallery_data(self, path):
        self.path = path
        self.gallery_data = ImageDataset(path, limit=None, frame_step=1)
        self.process_gallery_data()
        return self.gallery_data

    def process_gallery_data(self):
        if self.gallery_cache_builder is not None:
            gallery_cache_name = self.gallery_cache_builder("person", self.path)
        else:
            gallery_cache_name = os.path.abspath(self.path).replace("/", "_")
        
        print(gallery_cache_name)
        self.gallery = ReIdGallery(
            name=gallery_cache_name, 
            data=self.gallery_data, 
            data_extr=BoxExtractor(self.person_det, tracker=Tracker()), 
            feat_extr=self.feat_extr,
            gallery_dir="galleries/person/"
        )

    def create_query_db_from_image_files(self, img_files):
        query_db = ReIdGallery(
            data=ImageDataset(img_files),
            data_extr=BoxExtractorIdentity(), 
            # data_extr=BoxExtractor(person_det, single_box=True), 
            feat_extr=self.feat_extr
        )

        return query_db

    def query(self, query):
        query_results, distmat = self.gallery.query([query["feat"]], topk=self.topk, return_distmat=True)
        query_results = query_results[0]
        distmat = distmat[:1]
        return query_results, distmat
