import os
import sys
from types import SimpleNamespace
import cv2

from utils.mot.video_input import CapInput, VideoInput


class SimpleROMPRenderer:
    def __init__(self, romp_settings=None) -> None:
        romp_path = "../ROMP"

        sys.path.insert(0, romp_path)
        sys.path.insert(0, f"{romp_path}/simple_romp")
        from simple_romp.romp.main import ROMP as _ROMP

        if romp_settings is None:
            romp_settings = SimpleNamespace(
                mode="video", GPU=-1, onnx=True, temporal_optimize=False, center_thresh=0.25, show_largest=False, smooth_coeff=3.0, 
                calc_smpl=True, render_mesh=True, renderer='sim3dr', show=False, show_items='mesh',
                smpl_path=os.path.expanduser('~/.romp/smpl_packed_info.pth'), 
                model_path=os.path.expanduser('~/.romp/ROMP.pkl'), 
                model_onnx_path=os.path.expanduser('~/.romp/ROMP.onnx'), 
                root_align=False)

        self.romp = _ROMP(romp_settings)
        self.romp_path = romp_path

    def __call__(self, image):
        outputs = self.romp(image)
        outputs = outputs["rendered_image"][:, outputs["rendered_image"].shape[1] // 2:]
        return outputs


def _test():
    romp = SimpleROMPRenderer()

    cap = CapInput(f"{romp.romp_path}/demo/videos/sample_video.mp4")

    while cap.is_open():
        img = cap()
        outputs = romp(img)
        cv2.imshow("romp", outputs["rendered_image"])
        cv2.waitKey(1)


if __name__ == "__main__":
    _test()
