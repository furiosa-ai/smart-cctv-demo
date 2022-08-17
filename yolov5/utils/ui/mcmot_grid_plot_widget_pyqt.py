from collections import OrderedDict, namedtuple
import enum
import typing
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout

import pyqtgraph as pg

from utils.ui.mcmot_grid_plot_widget_base import MCMOTGridPlotWidgetBase


class MCMOTGridPlotWidgetPyQt(MCMOTGridPlotWidgetBase):
    def __init__(self, mot_count, parent=None) -> None:
        super().__init__(mot_count, parent)

        self.graph_view, self.plots = self._create_plot()

        vbox = QVBoxLayout()
        vbox.addWidget(self.graph_view)

        self.setLayout(vbox)

    def _create_plot(self):
        l = pg.GraphicsLayout()                                                                   

        plots = []
        for (x, y), title in zip(self.plot_coords, self.plot_titles):
            p = l.addPlot(y, x)
            p.setTitle(title)
            # p.setXRange(-5, 5)
            # p.setYRange(-5, 5)
            plots.append(p)

        # p0 = l.addPlot(0, 0)                                                                
        # p0.showGrid(x = True, y = True, alpha = 0.3)                                        
        #p0.hideAxis('bottom')                                                              
        # p1 = l.addPlot(1, 0)                                                                
        # p1.showGrid(x = True, y = True, alpha = 0.3)  

        view = pg.GraphicsView()                                                           
        view.setCentralItem(l)                                                              
        view.show()                                                                         
        view.resize(800,600)  

        return view, plots

    def update_plot(self, plot_name, lines, line_ids):
        plot = self._get_plot_by_name(plot_name)

        # plot.disableAutoRange()

        plot.clear()
        for line, line_id in zip(lines, line_ids):
            plot.plot(line[:, 0], line[:, 1], pen=pg.mkPen(color=self._id_to_color(line_id)))

        # plot.autoRange()