import numpy as np
import cv2


from vispy import scene, visuals
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout
from utils.logging import log_func
from utils.ui.plot_widget_vispy.plot_widget_vispy_3d_base import PlotWidgetVisPy3DBase
from utils.ui.cui_util import apply_line_color_gradient
from utils.util import PerfMeasure


class PlotWidgetVisPy3DLine(PlotWidgetVisPy3DBase):
    def __init__(self, plot_heads=True, plot_vert_lines=True, plot_axes=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.line_plot = scene.Line(width=0, parent=self.view.scene)  #  , method="agg" , antialias=True
        self.vert_line_width = 1
        self.line_width = 5
        # self.max_point_age = 100

        if plot_heads:
            Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
            self.heads = Scatter3D(size=0, symbol=None, parent=self.view.scene)
        else:
            self.heads = None

        if plot_vert_lines:
            self.vert_line = scene.Line(width=0, parent=self.view.scene)
        else:
            self.vert_line = None

        if plot_axes:
            x1, y1, x2, y2 = self.floor_img_xyxy
            z1, z2 = 0, 10

            # x1, y1, z1, x2, y2, z2 = 0, 0, 0, 1, 1, 1
            # x1, y1, z1, x2, y2, z2 = 0, 0, 0, 10, 10, 5
            # x1, y1, z1, x2, y2, z2 = -10, -10, 0, 10, 10, 5

            cx, cy, cz = "white", "white", "white"

            xax = scene.Axis(pos=[[x1, 0], [x2, 0]], domain=(x1, x2), tick_direction=(0, -1), axis_color=cx, tick_color=cx, text_color=cx, font_size=128, parent=self.view.scene)
            yax = scene.Axis(pos=[[0, y1], [0, y2]], domain=(y1, y2), tick_direction=(-1, 0), axis_color=cy, tick_color=cy, text_color=cy, font_size=128, parent=self.view.scene)

            zax = scene.Axis(pos=[[z1, 0], [-z2, 0]], domain=(z1, z2), tick_direction=(0, -1), axis_color=cz, tick_color=cz, text_color=cz, font_size=128, parent=self.view.scene)
            zax.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
            zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
            zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)

    @log_func
    def _plot(self, plot_data):
        if len(plot_data["connect"]) > 0:
            line_lengths = plot_data["line_lengths"]
            # point_age = plot_data["point_age"]
            # max_point_age = np.max(point_age)

            # color_mul = (max_point_age - point_age) / max_point_age
            # color_mul = 0.2 * color_mul + 0.8

            # color_mul = np.concatenate([np.linspace(0.8, 1, l) for l in line_lengths])
            # plot_data["color"] *= color_mul[:, None]
            plot_data["color"] = apply_line_color_gradient(plot_data)
            # plot_data["color"] = np.concatenate([plot_data["color"], color_mul[:, None]], 1)

            data = {
                "pos": plot_data["pos"],
                "connect": plot_data["connect"],
                "color": plot_data["color"],
                "width": self.line_width
            }
        else:
            data = { "width": 0 }

        self.line_plot.set_data(**data)

    @log_func
    def _create_heads_plot_data(self, plot_data):
        line_ids = plot_data["line_ids"]
        end_points = plot_data["end_points"]

        if len(line_ids) == 0:
            return None

        colors = self.color_table[line_ids % len(self.color_table)]

        pos = end_points

        return {
            "pos": pos,
            "color": colors
        }

    @log_func
    def _create_vert_line_plot_data(self, plot_data):
        line_ids = plot_data["line_ids"]
        end_points = plot_data["end_points"]

        if len(line_ids) == 0:
            return None

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
    def _plot_vert_line(self, plot_data):
        if plot_data is not None:
            data = {
                **plot_data,
                "width": self.vert_line_width
            }
        else:
            data = { "width": 0 }

        self.vert_line.set_data(**data)

    @log_func
    def _plot_heads(self, plot_data):
        if plot_data is not None:
            data = {
                "pos": plot_data["pos"],
                # "connect": plot_data["connect"],
                "face_color": plot_data["color"],
                "edge_color": plot_data["color"],
                    

                # "width": self.line_width
                "size": 20,
                "symbol": "star"
            }
        else:
            data = { 
                "pos": np.zeros((0,3)) 
            }
            # data = { "symbol": None }

        self.heads.set_data(**data, edge_width=1)

    @log_func
    def update_plot(self, plot_data):
        super().update_plot(plot_data)

        if self.vert_line is not None:
            plot_data_vert_line = plot_data["plot_data_vert_line"]
            self._plot_vert_line(plot_data_vert_line)

        if self.heads is not None:
            plot_data_heads = plot_data["plot_data_heads"]
            self._plot_heads(plot_data_heads)

    @log_func
    def create_plot_data(self, lines, line_ids, plot_state=None):
        plot_state = super().create_plot_data(lines, line_ids, plot_state=plot_state)

        # lines, line_ids, plot_data = super_plot_state["lines"], super_plot_state["line_ids"], super_plot_state["plot_data"]

        # assert not np.any(np.isnan(plot_data["pos"]))

        if "end_points" not in plot_state:
            end_point_idx = np.cumsum(plot_state["line_lengths"]) - 1
            # end_point_idx = np.cumsum(plot_data["line_lengths"][:-1])
            # end_point_idx = np.insert(end_point_idx, 0, 0)
            end_points = plot_state["pos"][end_point_idx]
            plot_state["end_points"] = end_points

        if self.vert_line is not None:
            if "plot_data_vert_line" not in plot_state:
                plot_state["plot_data_vert_line"] = self._create_vert_line_plot_data(plot_state)

        if self.heads is not None:
            if "plot_data_heads" not in plot_state:
                plot_state["plot_data_heads"] = self._create_heads_plot_data(plot_state)

        return plot_state
