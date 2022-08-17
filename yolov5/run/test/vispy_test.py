# -*- coding: utf-8 -*-
# vispy: testskip
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------


import cv2


import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QObject

import numpy as np
from vispy import app, gloo, scene


class MySignal(QObject):
    image = pyqtSignal(np.ndarray)

    def __init__(self) -> None:
        super().__init__()


class MyQThread(QThread):
    qt_signal = MySignal()

    def __init__(self, cfg=None, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg

    def run(self):
        cap = cv2.VideoCapture("datasets/test_vid/people_tracking0.mp4")

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                self.qt_signal.image.emit(frame)


class ImageViewCanvas(app.Canvas):
    def __init__(self):
        super().__init__(size=(640, 480), keys='interactive')

        vertex = """
            attribute vec2 position;
            attribute vec2 texcoord;
            varying vec2 v_texcoord;
            void main()
            {
                gl_Position = vec4(position, 0.0, 1.0);
                v_texcoord = texcoord;
            }
        """

        fragment = """
            uniform sampler2D texture;
            varying vec2 v_texcoord;
            void main()
            {
                gl_FragColor = texture2D(texture, v_texcoord).bgra;
            }
        """

        self.img = None
        
        self.program = gloo.Program(vertex, fragment, count=4)
        self.program['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.program['texcoord'] = [(1, 1), (1, 0), (0, 1), (0, 0)]
        self.program['texture'] = None

        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def update_image(self, img):
        self.img = img
        self.update()

    def on_draw(self, event):
        gloo.clear('black')

        if self.img is not None:
            self.program['texture'] = self.img
            self.program.draw('triangle_strip')


vertex = """
    attribute vec2 a_position;
    varying vec2 v_color;

    void main(){
        gl_Position = vec4(0.5 * a_position, 0.0, 1.0);
        v_color = a_position;
    }
"""

fragment = """
    varying vec2 v_color;
    void main()
    {
        gl_FragColor = vec4(0.0, 1.0, 1.0, 1.0);
        // gl_FragColor.g = 0.0;
        // gl_FragColor.b = 0.0;
        // gl_FragColor.a = 1.0;
    }
"""

class Canvas(app.Canvas):
    def __init__(self, show=False):
        super().__init__(size=(640, 480), keys='interactive')

        self.program = gloo.Program(vertex, fragment)
        self.program["a_position"] = np.array([
            [(-0.5, -1), (-1, +1), (+1, -1), (+1, +1)]
        ], dtype=np.float32)

        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        if show:
            self.show()

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, ev):
        gloo.clear('black')
        # return
        # gloo.set_viewport(0, 0, *self.size)  # hot fix
        self.program.draw("triangle_strip")

    def on_timer(self, event):
        self.update()


class MyPlotCanvas(scene.SceneCanvas):
    def __init__(self):
        super().__init__(size=(640, 480))

        view = self.central_widget.add_view()

        pos = np.array([[0,0], [1, 0], [0, 1], [1, 1]])
        pairlist = np.array([[0,1],[0,2], [0,3], [2,1], [2,3]])
        line = scene.Line(pos=pos, connect=pairlist, color=np.random.rand(4,3), width = 3, parent = view.scene)  
        

        view.camera = 'panzoom'

        # im = self.render()


class Widget(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet("color: green; background-color: green;")

        layout = QHBoxLayout()

        # w2 = MyCanvas()
        # w2 = Canvas()
        # w2 = MyPlotCanvas()
        self.image_widget = ImageViewCanvas()
        layout.addWidget(self.image_widget.native)

        self.setLayout(layout)

        self.thread = MyQThread()
        # connect its signal to the update_image slot
        self.thread.qt_signal.image.connect(self.update_image)
        # start the thread
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        self.image_widget.update_image(cv_img)


def run_qt():
    app = QApplication(sys.argv)
    a = Widget()
    a.show()
    sys.exit(app.exec_())



def run_vispy():
    # c = Canvas()
    c = MyCanvas(show=True)
    app.run()


def main():
    run_qt()


if __name__ == "__main__":
    main()
