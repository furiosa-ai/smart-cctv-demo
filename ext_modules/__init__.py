import sys
from pathlib import Path

import pandas
import mxnet
import torch
from ai_util.imp_env import ImpEnv
from ai_util.vision.image_gallery import QueryResult

root_dir = Path(__file__).parent.parent



with ImpEnv(root_dir / "yolov5"):
    from yolo_predictor.predictor import Yolov5Predictor
    from yolo_predictor.tracker import Tracker


with ImpEnv([root_dir / "insightface", root_dir / "insightface" / "recognition" / "arcface_torch"]):
    from recognition.arcface_torch.utils.arcface_predictor import ArcFacePredictor
    from recognition.arcface_torch.utils.face_extract import FaceExtractor
    from recognition.arcface_torch.utils.face_gallery import FaceGallery


with ImpEnv(root_dir / "torchreid"):
    from utils.reid_predictor import ReIdPredictor
    from utils.box_extract import BoxExtractor, BoxExtractorIdentity
    from utils.reid import ReIdGallery


with ImpEnv(root_dir / "yolov5_face"):
    from face_predictor.face_predictor import Yolov5FacePredictor

