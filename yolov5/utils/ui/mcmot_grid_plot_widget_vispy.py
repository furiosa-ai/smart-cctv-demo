from collections import OrderedDict, namedtuple
import enum
import numpy as np
import typing
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout

import pyqtgraph as pg
from utils.logging import log_func

from utils.ui.mcmot_grid_plot_widget_base import MCMOTGridPlotWidgetBase

from vispy import scene
from utils.ui.cui_util import create_plot_data
from utils.ui.plot_widget_vispy.plot_widget_vispy_2d import PlotWidgetVisPy2D
from utils.ui.plot_widget_vispy.plot_widget_vispy_3d_line import PlotWidgetVisPy3DLine
from utils.ui.plot_widget_vispy.plot_widget_vispy_3d_scatter import PlotWidgetVisPy3DScatter

from utils.util import PerfMeasure


class MCMOTGridPlotWidgetVisPy(MCMOTGridPlotWidgetBase):
    def __init__(self, main_window, mot_count, plot_type, show_sub_plot=None, parent=None, show_title=False, *args, **kwargs) -> None:
        super().__init__(main_window=main_window, mot_count=mot_count, parent=parent, show_sub_plot=show_sub_plot)

        self.setStyleSheet("color: white; background-color: black;")
        grid = QGridLayout()

        self.plots = []
        for i, (y, x) in enumerate(self.plot_coords):
            if plot_type == "2d":
                p = PlotWidgetVisPy2D(name=self.plot_titles[i], color_table=self.colors.as_table(), parent=self, *args, **kwargs)
            elif plot_type == "3d":
                p = PlotWidgetVisPy3DLine(name=self.plot_titles[i], color_table=self.colors.as_table(), parent=self, *args, **kwargs)
            else:
                raise Exception(f"Unknown plot type {p}")

            # p = PlotWidgetVisPy3DScatter(name=self.plot_titles[i], color_table=self.colors.as_table(), parent=self, *args, **kwargs)
            # p = PlotWidgetVisPy3DLine(name=self.plot_titles[i], color_table=self.colors.as_table(), parent=self, *args, **kwargs)
            
            if not show_title:
                grid.addWidget(p, y, x)
            else:
                lay = QVBoxLayout()
                lay.addWidget(QLabel("MOT 3d"))
                lay.addWidget(p)
                grid.addLayout(lay, y, x)

            self.plots.append(p)

        self.setLayout(grid)

    def autoPlotRange(self):
        for plot in self.plots:
            plot.autoPlotRange()

    def keyPressEvent(self, event):
        print('Key pressed')
        
    def create_plot_data(self, plot_name, *args, **kwargs):
        return self._get_plot_by_name(plot_name).create_plot_data(*args, **kwargs)

    def update_plot(self, plot_name, *args, **kwargs):
        return self._get_plot_by_name(plot_name).update_plot(*args, **kwargs)

