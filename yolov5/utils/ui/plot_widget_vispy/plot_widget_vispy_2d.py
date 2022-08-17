import numpy as np
import cv2


from vispy import scene, visuals
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QSizePolicy
from utils.logging import log_func
from utils.ui.plot_widget_vispy.plot_widget_vispy_base import PlotWidgetVisPyBase
from utils.ui.cui_util import create_plot_data
from utils.util import PerfMeasure


class PlotWidgetVisPy2D(PlotWidgetVisPyBase):
    def __init__(self, *args, **kwargs) -> None:
        canvas = scene.SceneCanvas(keys='interactive', size=(640, 480))

        super().__init__(*args, canvas=canvas, **kwargs)

        self.view = self._create_axes_view(canvas)

        self.line_plot = None

        """
        if self.show_point_count:
            point_count_label = QLabel("")
            point_count_label.setFixedSize(640, 30)
            vbox.addWidget(point_count_label)
        else:
            point_count_label = None

        self.point_count_label = point_count_label
        """

        self.last_min_pos = np.full(2, np.inf)
        self.last_max_pos = -np.full(2, np.inf)

        self.line_plot = scene.Line(width=0, parent=self.view.scene)
        self.line_width = 3

        if self.floor_img is not None:
            self._set_bg(self.floor_img, self.floor_img_xyxy)

    def _set_bg(self, floor_img, floor_img_xyxy):
        x1, y1, x2, y2 = floor_img_xyxy
        sx, sy = x2 - x1, y2 - y1

        bg_img = cv2.cvtColor(cv2.imread(floor_img), cv2.COLOR_BGR2RGB)
        bg_img //= 3
        bg = scene.Image(bg_img, parent=self.view.scene)
        bg.transform = visuals.transforms.STTransform(scale=(1/bg_img.shape[1] * sx, -1/bg_img.shape[0] * sy), translate=(x1, y2))

    def autoPlotRange(self):
        self.view.camera.set_range()

    def _create_simple_view(self):
        view = self.canvas.central_widget.add_view()
        view.camera = 'panzoom'

        return view

    def _create_axes_view(self, canvas):
        grid = canvas.central_widget.add_grid(margin=10)
        grid.spacing = 0

        # title = scene.Label(self.name, color='white')
        # title.height_max = 40
        title = scene.Label("", color='white')
        title.height_max = 0
        grid.add_widget(title, row=0, col=0, col_span=2)

        yaxis = scene.AxisWidget(orientation='left',
                                # axis_label='Y Axis',
                                axis_font_size=12,
                                axis_label_margin=50,
                                tick_label_margin=5)
        yaxis.width_max = 20
        grid.add_widget(yaxis, row=1, col=0)

        xaxis = scene.AxisWidget(orientation='bottom',
                                # axis_label='X Axis',
                                axis_font_size=12,
                                axis_label_margin=50,
                                tick_label_margin=5)

        xaxis.height_max = 20
        grid.add_widget(xaxis, row=2, col=1)

        right_padding = grid.add_widget(row=1, col=2, row_span=1)
        right_padding.width_max = 10

        view = grid.add_view(row=1, col=1, border_color='white')
        view.camera = 'panzoom'

        xaxis.link_view(view)
        yaxis.link_view(view)

        if self.plot_range is not None:
            x_range, y_range = self.plot_range
            view.camera.set_range(x_range, y_range)

        return view

    @log_func
    def _create_plot_data(self, lines, line_ids):
        return create_plot_data(self.color_table, lines, line_ids)
        """
        if np.any(min_pos < self.last_min_pos) or np.any(max_pos > self.last_max_pos):
            self.last_min_pos = np.minimum(self.last_min_pos, min_pos)
            self.last_max_pos = np.maximum(self.last_max_pos, max_pos)

            self.view.camera.set_range()
        """

    @log_func
    def _plot(self, plot_data):
        if len(plot_data["connect"]) > 0:
            pos = plot_data["pos"].copy()
            pos[:, 2] = -1

            data = {
                "pos": pos,
                "connect": plot_data["connect"],
                "color": plot_data["color"],
                "width": self.line_width
            }
        else:
            data = { "width": 0 }

        self.line_plot.set_data(**data)

    """
    @log_func
    def update_plot(self, lines, line_ids):
        num_lines = len(lines)

        with PerfMeasure("prep plot data"):
            plot_data = self._create_plot_data(lines, line_ids)

        with PerfMeasure("plot"):
            self._plot(plot_data)

        num_points = plot_data["pos"].shape[0]
        if self.point_count_label is not None:
            self.point_count_label.setText(f"#lines: {num_lines} #points: {num_points}")

        print(f"{self.name} #lines: {num_lines} #points: {num_points}")
    """