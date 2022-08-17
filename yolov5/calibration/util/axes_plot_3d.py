from re import L
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt

# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    ax.set_box_aspect([1, 1, 1])

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


class AxesPlot3d:
    fig = None
    cur_fig_idx = 1

    def __init__(self, num_row=1, num_col=1, transform_mat=None):
        if AxesPlot3d.fig is None:
            AxesPlot3d.fig = plt.figure()

        self.ax = self.fig.add_subplot(int(f"{num_row}{num_col}{AxesPlot3d.cur_fig_idx}"), projection='3d')
        AxesPlot3d.cur_fig_idx += 1

        self.y_up = False

        self.gl2mpl_mat = np.float32([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]).T if self.y_up else np.eye(4)  # in combination with ax.invert_yaxis() in show()

        self.transform_mat = None
        self.final_mat_t = None

        self.set_transform(transform_mat)

        self.graph = None

    def set_transform(self, transform_mat):
        if transform_mat is None:
            transform_mat = np.eye(4)

        self.transform_mat = transform_mat
        self.final_mat_t = np.matmul(self.gl2mpl_mat, self.transform_mat).T

    def _transform(self, x, y, z):
        p = np.stack([x, y, z, np.ones(len(x))], axis=1)
        p = np.matmul(p, self.final_mat_t)
        return p[:, 0], p[:, 1], p[:, 2]

    def plot_all(self, data, colors, **kwargs):
        for xyz, color in zip(data, colors):
            x, y, *z = xyz.T

            z = z[0] if len(z) > 0 else None

            self.plot(x, y, z, color=color, **kwargs)

    def plot(self, x, y, z=None, color=None, **kwargs):
        if z is None:
            z = np.zeros_like(x)

        p = self._transform(x, y, z)

        if self.graph is None:
            self.graph, = self.ax.plot(*p, color=color, **kwargs)
            self.graph = None
        else:
            self.graph.set_data(*p)
            # self.graph.set_xdata(p[0])
            # self.graph.set_ydata(p[1])
            # self.graph.set_3d_properties(p[2])
            self.graph.set_color(color)

    def scatter(self, x, y, z, **kwargs):
        self.ax.scatter(*self._transform(x, y, z), **kwargs)

    def clear(self):
        self.ax.clear()

    def show(self, block=False, pause=None):
        set_axes_equal(self.ax)

        """
        if self.y_up:
            self.ax.invert_yaxis()
        else:
            self.ax.invert_zaxis()
        """

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('z')
        self.ax.set_zlabel('y')
        plt.show(block=block)

        if pause is not None:
            plt.pause(pause)

    def save(self, filename):
        self.fig.savefig(filename)

    def draw(self):
        plt.draw()
