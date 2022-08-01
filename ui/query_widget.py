import sys


import time
import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QStyle, QSlider, QFileDialog
from PyQt5.QtGui import QPixmap
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from ui.select_file_widget import SelectFileWidget

from ui.ui_utils import convert_img_cv_to_qt


class _QueryWidget(QWidget):
    def __init__(self, preview_img):
        super().__init__()
        # create the label that holds the image
        # create a text label

        vbox = QVBoxLayout()
        self.image_view = QLabel()
        # self.image_view.resize(self.disply_width, self.display_height)
        vbox.addWidget(QLabel("Query"))
        vbox.addWidget(self.image_view)
        vbox.setAlignment(Qt.AlignTop)

        self.setLayout(vbox)

        self.image_view.setPixmap(convert_img_cv_to_qt(preview_img))


class QueryWidget(QWidget):
    query_changed = pyqtSignal(dict)

    def __init__(self, query_engine, file=None, parent=None) -> None:
        super().__init__(parent)

        self.query_engine = query_engine
        self.setFixedWidth(400)

        sel_file_widget = SelectFileWidget("Query", self._build_query_preview_widget, file=file)
        sel_file_widget.file_closed.connect(self.on_file_closed)

        lay = QVBoxLayout()
        lay.addWidget(sel_file_widget)
        self.setLayout(lay)

    def _build_query_preview_widget(self, file):
        query_db = self.query_engine.create_query_db_from_image_files([file])
        # query_db.data.open()
        query = query_db[0]
        preview_img = query_db.visualize(0)
        self.query_changed.emit(query)

        return _QueryWidget(preview_img)

    def on_file_closed(self):
        pass
