

from collections import OrderedDict, namedtuple
import enum
from math import ceil
import typing
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QCheckBox, QLineEdit
from PyQt5.QtGui import QPixmap
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QObject, QEvent
import numpy as np
from utils.fps_counter import FpsCounter
from utils.logging import logger, log_func
from utils.mot.mcmot_app_qt_proc import MCMOTPipelineProcessQThread
from utils.mot.mcmot_app_qt_utils import MCMOTQTSignal
from utils.mot.mcmot_display import MCMOTDisplay
from utils.colors import colors as _colors

from utils.mot.mcmot_pipeline import MCMOTPipeline
from utils.proctitle import set_proctitle
from utils.romp.romp_cpu_render_widget import ROMPCpuRenderWidget
from utils.ui.image_view_widget import ImageViewWidgetPyQt, ImageViewWidgetVisPy
from utils.ui.mcmot_grid_plot_widget_pyqt import MCMOTGridPlotWidgetPyQt
from utils.ui.mcmot_grid_plot_widget_vispy import MCMOTGridPlotWidgetVisPy

from pynput.keyboard import Key, Listener, KeyCode
from utils.ui.mot_view_widget import MOTViewWidget

from utils.util import MPLib, PerfMeasure
import setproctitle



class MCMOTQTDisplay(MCMOTDisplay):
    def __init__(self, qt_signal, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.qt_signal = qt_signal
        self.frame_idx = 0

        # assert self.record_vid is False, "Not working anymore, boxes are drawn in gl"

    def display_cameras(self, imgs, boxes):
        for i, (img, box) in enumerate(zip(imgs, boxes)):
            self.qt_signal.mot.emit(i, img, box)

    def _plot(self, plot_name, trajs, traj_ids):
        plot_data = self.qt_signal.parent.create_plot_data(plot_name, trajs, traj_ids)
        self.qt_signal.plot.emit(plot_name, plot_data)

    @log_func
    def draw_plot(self, mcmot, mot_cams):
        # if self.frame_idx % 2 != 0:
        #     return

        with PerfMeasure("Querying plot data"):
            for i in range(len(mot_cams)):
                plot_name = f"local_{i}"

                if plot_name in self.qt_signal.plot_names:
                    trajs_loc, traj_ids_loc = mcmot.query_trajs(mot_cams, track_type="local", color_by="global_id", cam_indices=[i])
                    self._plot(plot_name, trajs_loc, traj_ids_loc)

            if "local_all" in self.qt_signal.plot_names:
                trajs_loc_all, traj_ids_loc_all = mcmot.query_trajs(mot_cams, track_type="local", color_by="global_id")
                self._plot("local_all", trajs_loc_all, traj_ids_loc_all)

            if "global" in self.qt_signal.plot_names:
                trajs_glob, traj_ids_glob = mcmot.query_trajs(mot_cams, track_type="global", color_by="global_id")
                self._plot("global", trajs_glob, traj_ids_glob)

    def __call__(self, mcmot, mot_cams, mot_results):
        self.fps = self.fps_counter.step()
        print(f"FPS: {self.fps:.1f}")

        self.qt_signal.fps.emit(self.fps)

        if self.show_vid and self.show_plot:
            self.draw_plot(mcmot, mot_cams)

        if self.record_vid_raw:
            for cam_idx, mot_cam in enumerate(mot_cams):
                img = mot_results[cam_idx].image

                if self.record_vid_raw:
                    self._record_frame_raw(cam_idx, img)

        # will be drawn in gl now
        # if self.show_vid or self.record_vid:
        #     self.draw_boxes(mcmot, mot_cams, mot_results)

        if self.show_vid:
            imgs, boxes = zip(*[(res.image, mcmot.get_boxes(cam_idx, mot_cam)) for cam_idx, (res, mot_cam) in enumerate(zip(mot_results, mot_cams))])

            # self.draw_boxes(mcmot, mot_cams, mot_results)
            # imgs = self.imgs

            self.display_cameras(imgs, boxes)

        if self.record_vid:
            self.draw_boxes(mcmot, mot_cams, mot_results)
            self._make_grid(self.imgs)
            self._record_frame(self.img_grid)

        self.frame_idx += 1


class MCMOTPipelineQT(MCMOTPipeline):
    def __init__(self, cfg=None, qt_signal=None) -> None:
        super().__init__(cfg)
        self.qt_signal = qt_signal

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle qt_signal
        del state["qt_signal"]
        return state

    def _create_display(self):
        return MCMOTQTDisplay(mot_count=self.mot_count, qt_signal=self.qt_signal, **self.cfg["display"])


class MCMOTPipelineQThread(QThread):
    qt_signal = MCMOTQTSignal()

    def __init__(self, pipeline_cls=None, cfg=None, parent=None) -> None:
        super().__init__(parent)

        if pipeline_cls is None:
            pipeline_cls = MCMOTPipelineQT

        self.cfg = cfg
        self.pipeline_cls = pipeline_cls
        self.pipeline = None

    def run(self):
        # capture from web cam
        self.pipeline = self.pipeline_cls(self.cfg, qt_signal=self.qt_signal)
        self.pipeline.run()
        self.qt_signal.finished.emit()

    def exit_mcmot(self):
        self.pipeline.exit()


class MOTGridViewWidget(QWidget):
    def __init__(self, num_cams, num_cols, size_per_cam, show_track_ids):
        super().__init__()

        self.num_cams = num_cams
        self.num_cols = num_cols
        self.mot_layouts = None
        self.mot_widgets = None
        self.num_rows = ceil(num_cams / num_cols)
        self.size_per_cam = size_per_cam
        self.show_track_ids = show_track_ids
            
        self.initUI()

    def initUI(self):   
        grid = QGridLayout()  
        self.setLayout(grid)

        self.mot_layouts, self.mot_widgets = zip(*[self._create_mot_widget(i) for i in range(self.num_cams)])

        for i, l in enumerate(self.mot_layouts):
            # x = i % self.num_cols
            # y = i // self.num_cols
            x = i // self.num_cols
            y = i % self.num_cols
            grid.addLayout(l, y, x)

        # self.move(300, 150)
        # self.setWindowTitle('PyQt window')  
        # self.show()

    def _create_mot_widget(self, idx):
        vbox = QVBoxLayout()
        img_label = QLabel(f"Camera {idx}")

        # img_widget = ImageViewWidgetPyQt(size=self.size_per_cam)
        mot_widget = MOTViewWidget(size=self.size_per_cam, show_track_ids=self.show_track_ids)

        vbox.addWidget(img_label)
        vbox.addWidget(mot_widget)

        return vbox, mot_widget

    def set_data(self, idx, img, boxes):
        self.mot_widgets[idx].set_data(img, boxes)
    

class MCMOTSettingsWidget(QWidget):
    settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)

        self.settings = {
            # "auto_rotate": False
            "auto_rotate": 0
        }

        self.lay = QGridLayout()
        self.setLayout(self.lay)

        widgets = [
            self._create_textbox("Auto rotate", "auto_rotate")
        ]

        for i, (label, field) in enumerate(widgets):
            self.lay.addWidget(label, i, 0)
            self.lay.addWidget(field, i, 1)

        # self._create_checkbox("Auto Rotate", )

    def _update(self, func):
        func()
        self.settings_changed.emit(self.settings)

    def _update_key_val(self, key, val):
        self.settings[key] = val
        self.settings_changed.emit(self.settings)

    def _create_checkbox(self, label, key):
        b = QCheckBox(label)
        b.setChecked(self.settings[key])
        b.stateChanged.connect(lambda: self._update_key_val(key, b.isChecked()))
        return b

    def _create_textbox(self, label, key):
        label = QLabel(label)
        field = QLineEdit(str(self.settings[key]))
        field.textChanged.connect(lambda: self._update_key_val(key, field.text()))
        return label, field


class MCMOTApp(QWidget):    
    def __init__(self, cfg, pipeline_cls=None):
        super().__init__()

        MPLib.set_strategy(cfg["system"]["multiproc"])

        single_window = cfg["display"].get("single_window", False)

        self.setWindowTitle('MCMOT')
        # self.setWindowTitle("MCMOT")
        # self.disply_width = 640
        # self.display_height = 480
        # create the label that holds the image
        # self.image_label = QLabel(self)
        # self.graphWidget = pg.PlotWidget()
        # self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.setStyleSheet("color: white; background-color: black;")

        self.mot_count = len(cfg["mot"]["cameras"])
        self.plot_count = self.mot_count + 2
        self.plot_col_count = 2
        self.plot_row_count = ceil(self.plot_count / self.plot_col_count)

        self.mot_view_widget = MOTGridViewWidget(self.mot_count, self.plot_col_count, size_per_cam=cfg["display"]["size_per_mot"], 
            show_track_ids=cfg["display"].get("show_track_ids", True))
        # self.plot_widget = MCMOTGridPlotWidgetPyQt(self.mot_count)

        self.ui_fps_counter = FpsCounter()

        self.fps_label = QLabel()
        self.ui_fps_label = QLabel()

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.mot_view_widget)
        # hbox.addWidget(self.plot_widget)
        vbox.addLayout(hbox)
        vbox.addWidget(self.fps_label)
        vbox.addWidget(self.ui_fps_label)
        self.setLayout(vbox)

        self.settings_widget = MCMOTSettingsWidget()
        self.settings_widget.show()

        self.side_widgets = []

        if cfg["display"].get("show_plot", True):
            cfgs_plot = cfg["display"]["plots"]

            if not isinstance(cfgs_plot, (tuple, list)):
                cfgs_plot = [cfgs_plot]

            self.plot_widgets = [MCMOTGridPlotWidgetVisPy(main_window=self, mot_count=self.mot_count, show_title=i==0, **cfg_plot) 
                for i, cfg_plot in enumerate(cfgs_plot)]
        else:
            self.plot_widgets = []
        
        self.side_widgets += self.plot_widgets

        if "romp" in cfg["display"]:
            romp_cfg = cfg["display"]["romp"]
            romp_widget = ROMPCpuRenderWidget(romp_cfg)
            self.side_widgets.append(romp_widget)

        if "empty" in cfg["display"]:
            for text in cfg["display"]["empty"]:
                w = QLabel(text)
                w.setAlignment(Qt.AlignmentFlag.AlignTop)
                w.setFixedSize(640, 460)
                self.side_widgets.append(w)

        if single_window:
            self._add_side_widgets(hbox, self.side_widgets)
        else:
            for w in self.side_widgets:
                w.show()

        if cfg["mcmot"].get("as_sep_proc", True):
            th = MCMOTPipelineProcessQThread(cfg=cfg, pipeline_cls=pipeline_cls)
        else:
            th = MCMOTPipelineQThread(cfg=cfg, pipeline_cls=pipeline_cls)

        # create the video capture thread
        self.thread = th
        # connect its signal to the update_image slot
        self.thread.qt_signal.mot.connect(self.update_mot_view)
        self.thread.qt_signal.plot_names = self.plot_widgets[0].plot_names if len(self.plot_widgets) > 0 else [""]
        self.thread.qt_signal.plot.connect(self.update_plot)
        self.thread.qt_signal.fps.connect(self.update_fps)
        self.thread.qt_signal.finished.connect(self.mcmot_finished)
        self.thread.qt_signal.parent = self
        # start the thread
        self.thread.start()

        # self.key_monitor = KeyMonitor(self)
        # self.key_monitor.keyPressed.connect(self.keyPress)
        # self.key_monitor.start_monitoring()

        self.frame_idx = [0 for _ in range(self.mot_count)]

    def _add_side_widgets(self, lay, widgets):
        assert len(widgets) in (2, 3)

        pvbox = QVBoxLayout()
        pvbox.addWidget(widgets[0])
        if len(widgets) == 2:
            pvbox.addWidget(widgets[1])
        else:
            phbox = QHBoxLayout()
            phbox.addWidget(widgets[1])
            phbox.addWidget(widgets[2])
            pvbox.addLayout(phbox)
        lay.addLayout(pvbox)

    def settings_changed(self):
        return self.settings_widget.settings_changed

    def closeEvent(self, event):
        self.thread.exit_mcmot()

        print("Joining QThread")
        self.thread.wait()  


        print("Closing plot widget")
        for w in self.plot_widgets:
            w.close()
        self.settings_widget.close()

        print("Window closed")

    def create_plot_data(self, plot_name, lines, line_ids):
        plot_data = None
        for w in self.plot_widgets:
            plot_data = w.create_plot_data(plot_name, lines, line_ids, plot_state=plot_data)
        return plot_data

    @pyqtSlot()
    def mcmot_finished(self):
        self.close()

    @pyqtSlot(str, dict)
    def update_plot(self, plot_name, plot_data):
        for w in self.plot_widgets:
            w.update_plot(plot_name, plot_data)

        print("Updated plot")

    @pyqtSlot(int, np.ndarray, np.ndarray)
    def update_mot_view(self, idx, img, boxes):
        # if self.frame_idx[idx] % 2 == 0:
        self.mot_view_widget.set_data(idx, img, boxes)
        print(f"Updated camera view {idx}")

        self.frame_idx[idx] += 1

    @pyqtSlot(float)
    def update_fps(self, fps):
        fps_ui = self.ui_fps_counter.step()

        # self.fps_label.setText(f"FPS (MCMOT): {fps:.1f} | FPS (UI): {fps_ui:.1f}")
        self.fps_label.setText(f"FPS: {fps:.1f} | Inf/s: {4*fps:.1f}")
        # self.ui_fps_label.setText(f"UI FPS: {ui_fps:.1f}")
    
    """
    # @pyqtSlot(KeyCode)
    def keyPress(self, key):
        if key.char == "r":
            if self.plot_widget is not None:
                self.plot_widget.autoPlotRange()
    """


class KeyMonitor(QObject):
    keyPressed = pyqtSignal(KeyCode)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.listener = Listener(on_release=self.on_release)

    def on_release(self, key):
        self.keyPressed.emit(key)

    def stop_monitoring(self):
        self.listener.stop()

    def start_monitoring(self):
        self.listener.start()


"""
class MCMOTMainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setWindowTitle('MCMOT')
        
        # Cannot set QxxLayout directly on the QMainWindow
        # Need to create a QWidget and set it as the central widget
        self.widget = MCMOTApp(*args, **kwargs)
        self.setCentralWidget(self.widget) 

    def keyPressEvent(self, event):
        self.widget.fps_label.setText(f"key")
"""


def run_mcmot_qt(trace_code=False, *args, **kwargs):
    setproctitle.setproctitle("MCMOTPipelineQT")

    app = QApplication(sys.argv)
    a = MCMOTApp(*args, **kwargs)
    # a.showMaximized()
    a.show()
    code = app.exec_()

    # sys.exit(code)

