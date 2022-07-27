import sys

from ui.ui_utils import convert_img_cv_to_qt
print(sys.path)

import time
import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QStyle, QSlider
from PyQt5.QtGui import QPixmap
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np


class VideoThread(QThread):
    # stop_signal = pyqtSignal()
    frame_update_signal = pyqtSignal(int, int, np.ndarray)

    def __init__(self, src, parent=None) -> None:
        super().__init__(parent)

        self.is_playing = True
        self.src = src
        self.playback_pos_request = None

    def update_frame(self, image):
        # need previous frame index -> -1
        self.frame_update_signal.emit(self.src.get_frame_idx() - 1, self.src.get_frame_count(), image)

    def toggle_playback(self):
        self.is_playing = not self.is_playing

    def stop_playback(self):
        self.is_playing = False

    def set_playback_pos(self, pos):
        print(f"seek {pos}")
        self.playback_pos_request = pos

    def run(self):
        cur_frame = None

        self.src.open()

        while True:
            if self.playback_pos_request is not None or self.is_playing:
                if self.playback_pos_request is not None:
                    self.src.set_frame_idx(self.playback_pos_request)
                    self.playback_pos_request = None

                cur_frame = self.src.read()
                # cur_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            time.sleep(1 / self.src.get_fps())
            if cur_frame is not None:
                self.update_frame(cur_frame)


class VideoPlayerWidget(QWidget):
    def __init__(self, src):
        super().__init__()
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        button_box = QHBoxLayout()

        toggleButton = QPushButton()
        toggleButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        button_box.addWidget(toggleButton)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.set_position)
        button_box.addWidget(self.positionSlider)

        vbox.addWidget(self.image_label)
        vbox.addLayout(button_box)
        self.setLayout(vbox)

        # create the video capture thread
        self.video_playback = VideoThread(src)
        # connect its signal to the update_frame slot
        self.video_playback.frame_update_signal.connect(self.update_frame)
        # start the thread
        self.video_playback.start()
        
        toggleButton.clicked.connect(lambda: self.video_playback.toggle_playback())

    @pyqtSlot(int, int, np.ndarray)
    def update_frame(self, frame_idx, frame_count, cv_img):
        """Updates the image_label with a new opencv image"""

        self.positionSlider.blockSignals(True)
        self.positionSlider.setRange(0, frame_count - 1)
        self.positionSlider.setValue(frame_idx)
        self.positionSlider.blockSignals(False)

        qt_img = convert_img_cv_to_qt(cv_img, (self.disply_width, self.display_height))
        self.image_label.setPixmap(qt_img)

    def set_position(self, pos):
        self.video_playback.stop_playback()
        self.video_playback.set_playback_pos(pos)
        # self.mediaPlayer.setPosition(position)
