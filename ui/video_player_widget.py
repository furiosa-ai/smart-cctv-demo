import sys

from ui.ui_utils import convert_img_cv_to_qt
from utils.frame_rate import FrameRateSync
print(sys.path)

import time
import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QStyle, QSlider, QLineEdit
from PyQt5.QtGui import QPixmap, QIntValidator, QDoubleValidator
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np


class VideoThread(QThread):
    # stop_signal = pyqtSignal()
    frame_update_signal = pyqtSignal(int, int, np.ndarray)

    def __init__(self, src, fps, parent=None) -> None:
        super().__init__(parent)

        self.is_playing = True
        self.src = src
        self.playback_pos_request = None
        self.fps = fps
        self.fps_sync = FrameRateSync()

    def update_frame(self, image):
        # need previous frame index -> -1
        self.frame_update_signal.emit(self.src.get_frame_idx() - 1, self.src.get_frame_count(), image)

    def toggle_playback(self):
        self.is_playing = not self.is_playing

    def stop_playback(self):
        self.is_playing = False

    def set_playback_pos(self, pos):
        self.playback_pos_request = pos

    def redraw_frame(self):
        self.set_playback_pos(self.src.get_frame_idx() - 1)

    def set_fps(self, fps):
        self.fps = fps

    def run(self):
        cur_frame = None

        self.src.open()

        while True:
            self.fps_sync(self.fps)

            if self.playback_pos_request is not None or self.is_playing:
                if self.playback_pos_request is not None:
                    self.src.set_frame_idx(self.playback_pos_request)
                    self.playback_pos_request = None

                cur_frame = self.src.read()
                # cur_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            if cur_frame is not None:
                self.update_frame(cur_frame)


class PropertyWidget(QWidget):
    value_changed = pyqtSignal(str, object)

    def __init__(self, label, name, value, parent=None) -> None:
        super().__init__(parent)

        if isinstance(value, float):
            value_widget = QLineEdit(f"{value:.2f}")
            value_widget.setValidator(QDoubleValidator())
            value_widget.setFixedWidth(50)
            value_widget.textChanged.connect(lambda text: self.value_changed.emit(name, float(text)))
        else:
            raise Exception(value)

        self.value_widget = value_widget

        lay = QHBoxLayout()
        lay.addWidget(QLabel(label))
        lay.addWidget(value_widget)
        self.setLayout(lay)


class VideoPlayerWidget(QWidget):
    def __init__(self, src):
        super().__init__()

        fps = float(src.get_fps())

        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        control_layout = QVBoxLayout()
        slider_layout = QHBoxLayout()

        toggleButton = QPushButton()
        toggleButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        slider_layout.addWidget(toggleButton)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.set_position)
        slider_layout.addWidget(self.positionSlider)

        self.frame_idx_edit = QLineEdit()
        self.frame_idx_edit.setValidator(QIntValidator())
        self.frame_idx_edit.setFixedWidth(50)
        slider_layout.addWidget(self.frame_idx_edit)

        control_layout.addLayout(slider_layout)

        settings_layout = QHBoxLayout()

        fps_widget = PropertyWidget("FPS", "fps", fps)
        settings_layout.addWidget(fps_widget)

        control_layout.addLayout(settings_layout)

        vbox.addWidget(self.image_label)
        vbox.addLayout(control_layout)
        self.setLayout(vbox)

        # create the video capture thread
        self.video_playback = VideoThread(src, fps=fps)
        # connect its signal to the update_frame slot
        self.video_playback.frame_update_signal.connect(self.update_frame)
        # start the thread
        self.video_playback.start()
        
        toggleButton.clicked.connect(lambda: self.video_playback.toggle_playback())
        self.frame_idx_edit.textChanged.connect(lambda txt: self.video_playback.set_playback_pos(int(txt)))
        fps_widget.value_changed.connect(lambda _, new_fps: self.video_playback.set_fps(new_fps))

    def redraw_frame(self):
        self.video_playback.redraw_frame()

    @pyqtSlot(int, int, np.ndarray)
    def update_frame(self, frame_idx, frame_count, cv_img):
        """Updates the image_label with a new opencv image"""

        self.positionSlider.blockSignals(True)
        self.positionSlider.setRange(0, frame_count - 1)
        self.positionSlider.setValue(frame_idx)
        self.positionSlider.blockSignals(False)

        qt_img = convert_img_cv_to_qt(cv_img, (self.disply_width, self.display_height))
        self.image_label.setPixmap(qt_img)

        self.frame_idx_edit.setText(str(frame_idx))

    def set_position(self, pos):
        self.video_playback.stop_playback()
        self.video_playback.set_playback_pos(pos)
        # self.mediaPlayer.setPosition(position)
