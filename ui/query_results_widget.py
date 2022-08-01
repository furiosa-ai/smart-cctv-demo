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
from ai_util.vision.image_gallery import QueryResult

from ui.ui_utils import convert_img_cv_to_qt


class QueryResultsItemWidget(QListWidgetItem):
    def __init__(self, data, query_res):
        frame_idx = query_res.data_key
        ts_sec = (frame_idx / (data.get_frame_count() - 1)) * data.get_length_sec()
        time_str = str(datetime.timedelta(seconds=ts_sec))
        text = f"{time_str} ({query_res.dist:.2f})"

        super().__init__(text)

        self.query_res = query_res


class QueryResultsWidget(QWidget):
    result_selected = pyqtSignal(QueryResult)

    def __init__(self, query_engine):
        super().__init__()
        # create the label that holds the image
        # create a text label

        self.setFixedWidth(500)

        vbox = QVBoxLayout()

        self.listWidget = QListWidget()
        self.listWidget.itemClicked.connect(self.item_clicked)
        vbox.addWidget(self.listWidget)
        vbox.setAlignment(Qt.AlignTop)

        self.setLayout(vbox)

        self.query_engine = query_engine

    def item_clicked(self, item):
        self.result_selected.emit(item.query_res)

    def set_results(self, query_results):
        for query_res in query_results:
            self.listWidget.addItem(QueryResultsItemWidget(self.query_engine.gallery_data, query_res))

            # self.listWidget.addItem()