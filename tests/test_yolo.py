from ext_modules import Yolov5Predictor
import torch 
import numpy as np


def main():
    person_det = Yolov5Predictor(cfg="yolov5/models/yolov5m_warboy.yaml", weights="yolov5/runs/train/bytetrack_mot20_5data/weights/best.pt", 
            input_format="chw", input_prec="f32", calib_mode=None, quant_tag=None,
            input_size=(640, 640)).to("cpu")

    # x = torch.rand((1, 3, 640, 640))
    x = np.zeros((640, 640, 3), dtype=np.uint8)

    person_det([x])


if __name__ == "__main__":
    main()
