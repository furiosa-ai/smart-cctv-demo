import sys


import time
import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QStyle, QSlider, QFileDialog, QListWidget, QListWidgetItem
from PyQt5.QtGui import QPixmap
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import datetime
from ui.select_file_widget import SelectFileWidget

from ui.video_player_widget import VideoPlayerWidget


class QueryResultsItemWidget(QListWidgetItem):
    def __init__(self, data, query_res):
        frame_idx = query_res.data_key
        ts_sec = (frame_idx / (data.get_frame_count() - 1)) * data.get_length_sec()
        text = str(datetime.timedelta(seconds=ts_sec))

        super().__init__(text)

        self.query_res = query_res


class _GalleryWidget(QWidget):
    def __init__(self, query_engine, video_file):
        super().__init__()
        
        self.query_engine = query_engine
        self.query_engine.set_gallery_data(video_file)
        self.gallery_reader = query_engine.gallery.create_reader()
        self.player_widget = VideoPlayerWidget(self.gallery_reader)

        lay = QVBoxLayout()
        lay.addWidget(self.player_widget)
        self.setLayout(lay)

    def set_position(self, data_key):
        self.player_widget.set_position(data_key)

    def set_distmat(self, distmat):
        self.gallery_reader.set_distmat(distmat)


class GalleryWidget(QWidget):
    def __init__(self, query_engine, file=None, parent=None) -> None:
        super().__init__(parent)

        self.sel_file_widget = SelectFileWidget("Video", lambda file: _GalleryWidget(query_engine, file), file=file)

        lay = QVBoxLayout()
        lay.addWidget(self.sel_file_widget)
        self.setLayout(lay)

    def set_position(self, data_key):
        self.sel_file_widget.get_widget().set_position(data_key)

    def set_distmat(self, distmat):
        self.sel_file_widget.get_widget().set_distmat(distmat)
