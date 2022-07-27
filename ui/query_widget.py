import sys


import time
import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QStyle, QSlider, QFileDialog
from PyQt5.QtGui import QPixmap
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

from ui.ui_utils import convert_img_cv_to_qt


class QueryWidget(QWidget):
    query_changed = pyqtSignal(dict)

    def __init__(self, query_engine):
        super().__init__()
        # create the label that holds the image
        # create a text label

        self.query_engine = query_engine

        vbox = QVBoxLayout()
        self.label = QLabel(text="Query")
        self.image_view = QLabel()
        self.select_query = QPushButton(text="Open")
        self.select_query.clicked.connect(self._select_query)
        # self.image_view.resize(self.disply_width, self.display_height)

        vbox.addWidget(self.label)
        vbox.addWidget(self.image_view)
        vbox.addWidget(self.select_query)
        vbox.setAlignment(Qt.AlignTop)

        self.setLayout(vbox)

    def _select_query(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')

        if fname[0]:
            query_img_file = fname[0]

            query_db = self.query_engine.create_query_db_from_image_files([query_img_file])
            # query_db.data.open()
            query = query_db[0]
            preview_img = query_db.vis_box_lm(0)

            self.image_view.setPixmap(convert_img_cv_to_qt(preview_img))

            self.query_changed.emit(query)
