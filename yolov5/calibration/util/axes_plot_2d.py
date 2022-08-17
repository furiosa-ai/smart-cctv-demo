from re import L
import numpy as np  

import matplotlib.pyplot as plt


class AxesPlot2d:
    fig = None
    cur_fig_idx = 1

    def __init__(self, num_row=1, num_col=1, transform_mat=None):
        if AxesPlot2d.fig is None:
            AxesPlot2d.fig = plt.figure()

        self.ax = self.fig.add_subplot(int(f"{num_row}{num_col}{AxesPlot2d.cur_fig_idx}"), projection='3d')
        AxesPlot2d.cur_fig_idx += 1

        self.graph = None

    def plot_all(self, data, colors, **kwargs):
        for xy, color in zip(data, colors):
            x, y = xy.T
            self.plot(x, y, color=color, **kwargs)

    def plot(self, x, y, color=None, **kwargs):
        if self.graph is None:
            self.graph, = self.ax.plot(x, y, color=color, **kwargs)
            self.graph = None
        else:
            pass

    def clear(self):
        self.ax.clear()

    def show(self, block=False, pause=None):
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('z')
        plt.show(block=block)

        if pause is not None:
            plt.pause(pause)

    def save(self, filename):
        self.fig.savefig(filename)

    def draw(self):
        plt.draw()
