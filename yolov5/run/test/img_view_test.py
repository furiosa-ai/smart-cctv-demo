from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from vispy import app, gloo, visuals, scene
import cv2
import numpy as np
from PyQt5 import QtCore
import sys
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QMainWindow

from utils.ui.image_view_widget import ImageView
from utils.ui.mot_view_widget import MOTViewWidget
from utils.ui.ui_util import create_box_data


from utils.colors import colors

 

class TestWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.img_view = MOTViewWidget(size=(640, 480))

        vbox = QVBoxLayout()
        vbox.addWidget(self.img_view)
        self.setLayout(vbox)

        tex = cv2.imread("cam_calib_data/cam_single_test/cam_calib0/640_480/extr_img.png")

        boxes = np.array([
            [0.2, 0.3, 0.4, 0.7, 0]
            # [0.3, 0.5, 0.7, 0.9, 0],
        ], dtype=np.float32)

        self.img_view.set_data(tex, boxes)


def main(*args, **kwargs):
    app = QApplication(sys.argv)
    a = TestWidget(*args, **kwargs)
    a.show()
    code = app.exec_()

main()
