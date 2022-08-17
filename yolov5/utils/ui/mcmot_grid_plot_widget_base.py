
from collections import OrderedDict
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout

from math import ceil

from utils.colors import colors as _colors


class MCMOTGridPlotWidgetBase(QWidget):
    def __init__(self, main_window, mot_count, show_sub_plot=None, parent=None) -> None:
        super().__init__(parent)

        self.mot_count = mot_count
        self.plot_col_count = 2
        self.colors = _colors

        self.plot_names = []
        self.plot_titles = []

        if show_sub_plot is None:
            show_sub_plot = { "local": range(mot_count), "local_all": True, "global": True }

        for i in range(self.mot_count):
            if i in show_sub_plot["local"]:
                self.plot_names.append(f"local_{i}")
                self.plot_titles.append(f"Camera {i}")

        if show_sub_plot["local_all"]:
            self.plot_names.append(f"local_all")
            self.plot_titles.append(f"All Cameras")

        if show_sub_plot["global"]:
            self.plot_names.append(f"global")
            self.plot_titles.append(f"MCMOT")

        self.plot_count = len(self.plot_names)
        self.plot_row_count = ceil(self.plot_count / self.plot_col_count)
        self.plot_coords = []
        for i in range(self.plot_count):
            x = i % self.plot_col_count
            y = i // self.plot_col_count
            self.plot_coords.append((x, y))

        self.plot_names_coords = OrderedDict(zip(self.plot_names, self.plot_coords))
        self.main_window = main_window

    def settings_changed(self):
        return self.main_window.settings_changed()

    def _get_plot_by_name(self, name):
        return self.plots[list(self.plot_names_coords.keys()).index(name)]

    def _id_to_color(self, id):
        return self.colors(id) 

    def update_plot(self, plot_name, lines, line_ids):
        raise NotImplementedError()
