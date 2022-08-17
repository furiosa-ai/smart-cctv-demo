import numpy as np
import cv2
import time

from vispy import scene, visuals
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5.QtCore import QTimer
from utils.logging import log_func
from utils.ui.plot_widget_vispy.plot_widget_vispy_3d_base import PlotWidgetVisPy3DBase
from utils.ui.cui_util import create_plot_data
from utils.util import PerfMeasure


class PlotWidgetVisPy3DScatter(PlotWidgetVisPy3DBase):
    def __init__(self, plot_vert_lines=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        self.plt = Scatter3D(parent=self.view.scene)

        if plot_vert_lines:
            self.line = scene.Line(width=0, parent=self.view.scene)
            self.line_width = 3
        else:
            self.line = None

        # self.timer = QTimer(parent=self)
        # self.timer.timeout.connect(self._timer_tick)
        # self.timer.start(1 / 30)

    def _create_vert_line_plot_data(self, lines, line_ids):
        end_points = np.array([l[-1] for l in lines])
        colors = self.color_table[line_ids % len(self.color_table)]

        num_points = len(end_points) * 2

        ground_points = end_points.copy()
        ground_points[:, 2] = 0

        pos = np.stack([ground_points, end_points], 1)
        pos = pos.reshape(num_points, 3)
        colors = colors.repeat(2, 0)

        con = np.arange(num_points).reshape(len(end_points), 2)

        return {
            "pos": pos,
            "connect": con,
            "color": colors
        }

    @log_func
    def _plot(self, plot_data):
        if len(plot_data["connect"]) > 0:

            line_lengths = plot_data["line_lengths"]
            marker_size = np.concatenate([np.linspace(0, 10, l) for l in line_lengths])

            data = {
                "pos": plot_data["pos"],
                # "connect": plot_data["connect"],
                "face_color": plot_data["color"],
                "edge_color": plot_data["color"],
                 

                # "width": self.line_width
                "size": marker_size
            }

            self.plt.set_data(**data, symbol="o", edge_width=1, edge_width_rel=None)
            # self.plt.set_gl_state('translucent', blend=True, depth_test=True)
            # self.plt.symbol = visuals.marker_types[10]
        else:
            data = { 
                # "width": 0 
            }

    @log_func
    def _plot_line(self, plot_data):
        if plot_data is not None:
            data = {
                **plot_data,
                "width": self.line_width
            }
        else:
            data = { "width": 0 }

        self.line.set_data(**data)

    @log_func
    def update_plot(self, lines, line_ids):
        lines, line_ids, plot_data = super().update_plot(lines, line_ids)

        if self.line is not None:
            if  len(lines) > 0:
                plot_data_line = self._create_vert_line_plot_data(lines, line_ids)

                with PerfMeasure("plot"):
                        self._plot_line(plot_data_line)
