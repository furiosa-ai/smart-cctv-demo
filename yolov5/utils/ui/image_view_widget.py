
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from vispy import app, gloo, visuals, scene
import cv2
import typing
from PyQt5 import QtCore
import numpy as np


class ImageViewWidgetPyQt(QLabel):
    def __init__(self, size):
        super().__init__()

        self.setFixedWidth(size[0])
        self.setFixedHeight(size[1])

    def update_image(self, img):
        qt_img = self.convert_cv_qt(img)
        self.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format
        return QPixmap.fromImage(p)



class ImageViewWidgetVisPy(QWidget):
    def __init__(self, size, parent=None) -> None:
        super().__init__(parent=parent)

        self.canvas = scene.SceneCanvas(size=size)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas.native)
        self.setLayout(vbox)

        self.img_view = ImageView(parent=self.canvas.scene)

    def update_image(self, img):
        self.img_view.set_data(img)


class ImageViewVisual(visuals.Visual):
    def __init__(self):
        vertex_shader = """
            varying vec2 v_texcoord;
            void main() {
                gl_Position = vec4($position, 0, 1);
                v_texcoord = $texcoord;
            }
        """

        fragment_shader = """
            varying vec2 v_texcoord;
            uniform sampler2D texture;

            void main() {
                gl_FragColor = texture2D(texture, v_texcoord).bgra;
            }
        """

        super().__init__(vertex_shader, fragment_shader)

        self.img = None
        self.need_img_upload = False

        self.vbo = gloo.VertexBuffer(np.array([
            (-1, -1), (-1, +1), (+1, -1), (+1, +1)
        ], dtype=np.float32))

        self.vbo_tex = gloo.VertexBuffer(np.array([
            [(1, 1), (1, 0), (0, 1), (0, 0)]
        ], dtype=np.float32))

        self.shared_program.vert['position'] = self.vbo
        self.shared_program.vert['texcoord'] = self.vbo_tex
        self.shared_program['texture'] = None
        self._draw_mode = 'triangle_strip'

    def set_data(self, img):
        self.img = img
        self.need_img_upload = True
        
    def _prepare_transforms(self, view):
        pass

    def _prepare_draw(self, view):
        if self.img is None:
            return False

        if self.need_img_upload:
            self.shared_program['texture'] = self.img
            self.need_img_upload = False


ImageView = scene.visuals.create_visual_node(ImageViewVisual)


class _ImageViewWidgetVisPyRaw(app.Canvas):
    def __init__(self, size):
        super().__init__(size=size, keys='interactive')

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

