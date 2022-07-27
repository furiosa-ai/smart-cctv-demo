import sys
from ui.query_results_widget import QueryResultsWidget
from ui.query_widget import QueryWidget

from ui.video_player_widget import VideoPlayerWidget
from utils.query_engine_dummy import QueryEngineDummy
from utils.query_engine_face import QueryEngineFace
print(sys.path)

import time
import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QStyle, QSlider, QFileDialog
from PyQt5.QtGui import QPixmap
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np


class App(QWidget):
    def __init__(self, query_engine, video_file=None):
        super().__init__()
        self.setWindowTitle("Smart CCTV Demo")

        # src = "/Users/kevin/Documents/projects/data/test_face/tc1/tom_cruise_test.mp4"

        self.query_engine = query_engine
        self.hbox = QHBoxLayout()

        # self.open_file_button = QPushButton("Open video")
        # self.open_file_button.clicked.connect(self.open_video_file)
        # self.hbox.addWidget(self.open_file_button)

        if video_file is not None:
            # self.open_video_file(video_file)
            self.query_engine.set_gallery_data(video_file)

            query_widget = QueryWidget(query_engine)
            query_widget.query_changed.connect(self.query_changed)

            self.query_res_widget = QueryResultsWidget(query_engine.gallery_data)
            # self.player_widget = VideoPlayerWidget(query_engine.gallery_data)
            self.gallery_reader = query_engine.gallery.create_reader()
            self.player_widget = VideoPlayerWidget(self.gallery_reader)
            self.query_res_widget.result_selected.connect(self.result_selected)

            self.hbox.addWidget(query_widget)
            self.hbox.addWidget(self.player_widget)
            self.hbox.addWidget(self.query_res_widget)

        # vbox.addWidget(VideoPlayerWidget(src))
        self.setLayout(self.hbox)

    def result_selected(self, query_res):
        self.player_widget.set_position(query_res.data_key)

    def query_changed(self, query):
        results, distmat = self.query_engine.query(query)
        self.gallery_reader.set_distmat(distmat)
        self.query_res_widget.set_results(results)

    def open_video_file(self, fname=None):
        if fname is None:
            fname = QFileDialog.getOpenFileName(self, 'Open file', './')

            if not fname[0]:
                return
            else:
                fname = fname[0]

        self.open_file_button.setParent(None)

        self.query_engine.set_gallery_data(fname)
        self.hbox.addWidget(VideoPlayerWidget(self.query_engine.gallery.create_reader()))
    

def main():
    # query_engine = QueryEngineDummy()
    query_engine = QueryEngineFace()
    video_file = "/Users/kevin/Documents/projects/data/test_face/tc1/tom_cruise_test.mp4"

    app = QApplication(sys.argv)
    a = App(query_engine, video_file)
    a.show()
    sys.exit(app.exec_())


if __name__=="__main__":
    main()