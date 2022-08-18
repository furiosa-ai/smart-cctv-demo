
import argparse
from functools import partial
import sys
import os
import yaml

sys.path.insert(0, os.getcwd())

import ext_modules

# import importlib
# from utils.query_engine_dummy import QueryEngineDummy
from utils.query_engine_reid import QueryEngineReId
from utils.query_engine_face import QueryEngineFace
from utils.query_engine_base import GalleryCacheBuilderRemote

from collections import OrderedDict
import sys
from ui.gallery_widget import GalleryWidget
from ui.query_results_widget import QueryResultsWidget
from ui.query_widget import QueryWidget
from ui.select_file_widget import SelectFileWidget

from ui.video_player_widget import VideoPlayerWidget
print(sys.path)

import time
import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QStyle, QComboBox, QFileDialog, QCheckBox
from PyQt5.QtGui import QPixmap
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np


class VideoAnalyzeWidget(QWidget):
    settings_changed = pyqtSignal()

    def __init__(self, query_engine_builder):
        super().__init__()
        self.setWindowTitle("Smart CCTV Demo")

        query_engine = query_engine_builder()
        self.query_engine = query_engine
        self.settings = {}

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()

        query_widget = QueryWidget(query_engine)
        query_widget.query_changed.connect(self.query_changed)

        self.gallery_widget = GalleryWidget(self, file=None)

        self.query_res_widget = QueryResultsWidget(query_engine)
        self.query_res_widget.result_selected.connect(self.result_selected)

        hbox.addWidget(query_widget)
        hbox.addWidget(self.gallery_widget)
        hbox.addWidget(self.query_res_widget)

        settingsWidget = QHBoxLayout()

        show_best_only_check = QCheckBox("Best match only")
        # show_best_only_check.stateChanged(lambda state: self.gallery_widget.update_settings(vis_best_only=state != 0))
        show_best_only_check.stateChanged.connect(lambda state: self._settings_changed(vis_best_only=state != 0))
        settingsWidget.addWidget(show_best_only_check)

        # vbox.addWidget(VideoPlayerWidget(src))
        vbox.addLayout(hbox)
        vbox.addLayout(settingsWidget)
        self.setLayout(vbox)

    def _settings_changed(self, **kwargs):
        for key, value in kwargs.items():
            self.settings[key] = value
        self.settings_changed.emit()

    def result_selected(self, query_res):
        self.gallery_widget.set_position(query_res.data_key)

    def query_changed(self, query):
        results, distmat = self.query_engine.query(query)
        self.gallery_widget.set_distmat(distmat)
        self.query_res_widget.set_results(results)
    

class App(QWidget):
    def __init__(self, args, parent=None) -> None:
        super().__init__(parent)

        if args.server is not None:
            with open(args.server, "r") as f:
                server_cfg = yaml.safe_load(f)

            gallery_cache_builder = GalleryCacheBuilderRemote(**server_cfg)
        else:
            gallery_cache_builder = None

        query_engine_args = dict(
            gallery_cache_builder=gallery_cache_builder,
        )

        self.cur_widget = None
        self.analyze_widgets = [
            ("Face", VideoAnalyzeWidget(partial(QueryEngineFace, **query_engine_args))),
            ("Person", VideoAnalyzeWidget(partial(QueryEngineReId, **query_engine_args)))
        ]

        names, _ = zip(*self.analyze_widgets)

        combo_box = QComboBox()
        combo_box.addItems(names)
        combo_box.currentIndexChanged.connect(self.selection_changed)

        self.lay = QVBoxLayout()
        self.lay.addWidget(combo_box)
        self.setLayout(self.lay)

        self.selection_changed(0)

    def selection_changed(self, idx):
        if self.cur_widget is not None:
            self.lay.removeWidget(self.cur_widget)
            self.cur_widget.setParent(None)
            # TODO: also free warboy

        self.cur_widget = self.analyze_widgets[idx][1]
        self.lay.addWidget(self.cur_widget)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="cfg/server_vis.yaml")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    a = App(args)
    a.show()
    sys.exit(app.exec_())


if __name__=="__main__":
    main()