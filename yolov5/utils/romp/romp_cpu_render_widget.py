import sys
import numpy as np

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QObject, QEvent
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QCheckBox, QLineEdit
from utils.mot.video_input import VideoInput
from vispy import app, gloo, visuals, scene

from utils.ui.image_view_widget import ImageView
from utils.util import MPLib


class ROMPProcess:
    def __init__(self, cfg, out_qu):

        import sys
        from utils.romp.romp_renderer import ROMPRenderer
        self.out_qu = out_qu

        self.romp = ROMPRenderer()
        self.video_input = VideoInput(cfg["input"]["src"], size=cfg["input"]["size"])
        assert self.video_input.is_open()

    def run(self):
        while self.video_input.is_open():
            img = self.video_input()
            output_img = self.romp(img)
            self.out_qu.put([output_img])

    @staticmethod
    def get_output_format(cfg):
        cam_res = cfg["input"]["size"]
        return [np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)], [True]

    @staticmethod
    def create_run(*args, **kwargs):
        proc = ROMPProcess(*args, **kwargs)
        proc.run()


class ROMPMessageThread(QThread):
    romp_signal = pyqtSignal(np.ndarray)

    def __init__(self, cfg, use_shm=True, parent=None):
        super().__init__(parent)

        self.cfg = cfg
        self.use_shm = use_shm
        self.is_running = True

    def msg_loop(self, qu):
        while self.is_running:
            img, = qu.get()
            self.romp_signal.emit(img)

    def run(self) -> None:
        if self.use_shm:
            fmt, shm = ROMPProcess.get_output_format(self.cfg)
            Qu = MPLib.create_dyn_queue(fmt, shm)
        else:
            Qu = MPLib.Queue

        qu = Qu(1)

        proc = MPLib.Process(target=ROMPProcess.create_run, kwargs=dict(cfg=self.cfg, out_qu=qu))
        proc.start()
        self.msg_loop(qu)
        proc.join()


class ROMPCpuRenderWidget(QWidget):
    def __init__(self, cfg, parent=None):
        super().__init__(parent=parent)

        size = cfg.get("widget_size", [640, 360])

        self.canvas = scene.SceneCanvas(size=size)
        self.img_view = ImageView(parent=self.canvas.scene)

        lay = QVBoxLayout()
        lay.addWidget(QLabel("ROMP"))
        lay.addWidget(self.canvas.native)
        self.setLayout(lay)

        self.thread = ROMPMessageThread(cfg=cfg, parent=self)
        self.thread.romp_signal.connect(self.update_romp)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_romp(self, img):
        self.img_view.set_data(img)
        self.img_view.update()


def _test():
    MPLib.set_strategy("process")

    cfg = {
        "input": {
            "src": "../ROMP/demo/videos/sample_video.mp4",
            "size": [1280, 720],
        }
    }

    app = QApplication(sys.argv)
    a = ROMPCpuRenderWidget(cfg=cfg)
    a.show()
    code = app.exec_()


if __name__ == "__main__":
    _test()
