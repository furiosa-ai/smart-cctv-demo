


import torchreid
from torchreid.data.transforms import build_transforms
from torchreid.utils import load_pretrained_weights
from utils.inference_framework import PredictorBase


"""
class ReIdPredictor(PredictorBase):
    def __init__(self, name, engine, input_size, preproc_input=False, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.input_size = input_size
        self.pytorch_model = engine.model
        self.calib_dataset = engine.dataset
        self.preproc_transform = build_transforms(width=self.input_size[0], height=self.input_size[1])[1]
        self.preproc_input = preproc_input

    def preproc(self, img, input_format, input_prec):
        if self.preproc_input:
            x = self.preproc_transform(img)
        else:
            x = img

        return x, None

    def postproc(self, pred, info):
        return pred

    def build_model(self):
        return self.pytorch_model

    def to(self, device, out_file=None):
        self.get_model()
        super().to(device, out_file)

    def get_calibration_dataset(self):
        return self.calib_dataset
"""


def main():
    infer_engine = OnnxInferenceEngineI8()


if __name__ == "__main__":
    main()
