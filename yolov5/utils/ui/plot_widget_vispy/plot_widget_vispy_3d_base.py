import numpy as np
import cv2
import time


from vispy import scene, visuals
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout
from utils.logging import log_func
from utils.ui.plot_widget_vispy.plot_widget_vispy_base import PlotWidgetVisPyBase
from utils.ui.cui_util import create_plot_data
from utils.util import PerfMeasure


class PlotWidgetVisPy3DBase(PlotWidgetVisPyBase):
    def __init__(self, parent, *args, **kwargs) -> None:
        assert parent is not None
        canvas = scene.SceneCanvas(keys='interactive', size=(640, 480))

        super().__init__(*args, canvas=canvas, parent=parent, **kwargs)

        view = canvas.central_widget.add_view()
        view.camera = 'turntable'

        self.view = view

        if self.floor_img is not None:
            self._set_bg(self.floor_img, self.floor_img_xyxy)

    @log_func
    def _move_scene(self, delta_time):
        if self.auto_rotate > 0:
            self.view.camera.transform.rotate(self.auto_rotate * delta_time, [0, 0, 1])

    def _set_bg(self, floor_img, floor_img_xyxy):
        x1, y1, x2, y2 = floor_img_xyxy
        sx, sy = x2 - x1, y2 - y1

        bg_img = cv2.imread(floor_img)
        bg_img = cv2.cvtColor(cv2.imread(floor_img), cv2.COLOR_BGR2RGB)
        # bg_img //= 2
        bg = scene.Image(bg_img, parent=self.view.scene)
        bg.transform = visuals.transforms.STTransform(scale=(1/bg_img.shape[1] * sx, -1/bg_img.shape[0] * sy), translate=(x1, y2))
