import sys

from utils.mot.video_input import CapInput
print(sys.path)

import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QMainWindow, QDesktopWidget
from PyQt5.QtGui import QPixmap
import sys
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QPoint
import numpy as np


from multiprocessing import Process


class VideoThreadBase(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def update_image(self, image):
        self.change_pixmap_signal.emit(image)


class VideoThread(VideoThreadBase):
    def run(self):
        # capture from web cam

        # cap = cv2.VideoCapture(0)
        cap = CapInput("datasets/MOT20/vid/MOT20-02-640.mp4", loop_video=True)
        while True:
            cv_img = cap()
            self.update_image(cv_img)
            # cv2.waitKey(20)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt live label demo")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.img_width = 640
        self.img_size = 360
        self.display_width = 340
        self.display_height = 360
        # create the label that holds the image


        self.image_label = QLabel(self)
        # self.image_label.setFixedSize(640, 360)
        # self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Webcam')

        # create a vertical box layout and add the two labels
        # vbox = QVBoxLayout()
        # vbox.addWidget(self.image_label)
        # vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        # self.setLayout(vbox)

        self.setCentralWidget(self.image_label)

        self.resize(self.display_width, self.display_height)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        # h, w = cv_img.shape[:2]
        x1 = (self.img_width - self.display_width) // 2
        x2 = self.img_width - self.display_width - x1

        cv_img = cv_img[:, x1:-x2]

        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        # self.image_label.setScaledContents(True)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint (event.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()
    

if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
