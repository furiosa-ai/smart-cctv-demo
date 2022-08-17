from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout
from utils.logging import log_func

from utils.util import PerfMeasure
from utils.ui.cui_util import create_plot_data

import time
import numpy as np
from scipy.ndimage import convolve1d


class PlotWidgetVisPyBase(QWidget):
    def __init__(self, canvas, name=None, parent=None, color_table=None, plot_range=((-10, 10), (-10, 10)), show_point_count=True, floor_img=None, floor_img_xyxy=None, line_smoothing=False, *args, **kwargs) -> None:
        super().__init__(parent)

        self.canvas = canvas
        # self.view = view
        self.name = name
        self.color_table = color_table
        self.plot_range = plot_range
        self.show_point_count = show_point_count
        self.floor_img = floor_img
        self.floor_img_xyxy = floor_img_xyxy
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas.native)
        self.setLayout(vbox)
        
        self.ts = None

        # plane = scene.visuals.Plane(parent=view.scene, width=20, height=20)

        self.ts = None
        self.auto_rotate = 5
        self.line_smoothing = line_smoothing

        self.window().settings_changed().connect(self.settings_changed)

    @log_func
    def transform_plot_data(self, plot_data):
        pos = plot_data["pos"]
        pos[:, 1] = -pos[:, 1]

        if pos.shape[1] == 3:
            pos[:, 2] *= 0.5

        plot_data["pos"] = pos
        return plot_data

    # def _set_bg(self, floor_img, floor_img_xyxy):
    #     raise NotImplementedError()

    def settings_changed(self, settings):
        try:
            auto_rotate = float(settings["auto_rotate"])
        except ValueError:
            auto_rotate = 0

        self.auto_rotate = auto_rotate

    # def _timer_tick(self):
    #     self.view.camera.transform.rotate(0.1, [0, 0, 1])

    def _move_scene(self, delta_time):
        pass
  
    @log_func
    def _create_plot_data(self, lines, line_ids):
        return create_plot_data(self.color_table, lines, line_ids)

    def _plot(self, plot_data):
        raise NotImplementedError()

    @log_func
    def _smooth_lines(self, lines, line_ids):
        # smooth lines
        if self.line_smoothing and len(lines) > 0:
            kernel_size = 5
            clip = kernel_size // 2
            smooth_kernel = np.float64([1 / kernel_size] * kernel_size)
            lines_smooth = []
            line_ids_smooth = []
            
            for line, line_id in zip(lines, line_ids):
                if line.shape[0] >= kernel_size:
                    lines_smooth.append(convolve1d(line, smooth_kernel, axis=0)[clip:-clip])
                    line_ids_smooth.append(line_id)

            lines = lines_smooth
            line_ids = np.array(line_ids_smooth, dtype=line_ids.dtype)

        return lines, line_ids

    @log_func
    def update_plot(self, plot_data):
        cur_time = time.time()

        if self.ts is None:
            self.ts = cur_time
        
        self._move_scene(cur_time - self.ts)
        self.ts = cur_time

        with PerfMeasure("plot"):
            self._plot(plot_data)

    @log_func
    def create_plot_data(self, lines, line_ids, plot_state=None):
        if plot_state is None:
            plot_state = {}

        if "lines" not in plot_state:
            plot_state["lines"], plot_state["line_ids"] = self._smooth_lines(lines, line_ids)

        if "pos" not in plot_state:
            plot_state = self.transform_plot_data(self._create_plot_data(lines, line_ids))

        return plot_state
