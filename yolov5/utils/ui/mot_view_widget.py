import enum
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from vispy import visuals, scene
import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from utils.logging import log_func

from utils.ui.image_view_widget import ImageView
from utils.ui.ui_util import create_box_data


from utils.colors import colors
from utils.util import PerfMeasure


class BoxVisual(visuals.LineVisual):
    def __init__(self, size, line_width=3, *args, **kwargs):
        super().__init__(width=0, *args, **kwargs)

        self.unfreeze()
        self.line_width = line_width
        self.color_table = colors.as_table()
        self.w = size[0]
        self.h = size[1]
        self.freeze()

    @log_func
    def set_data(self, boxes):
        # x needs to be mirrored (dont know why)
        if boxes is not None and len(boxes) > 0:
            boxes[:, [0, 2]] = self.w - boxes[:, [0, 2]]
            box_data = {
                **create_box_data(self.color_table, boxes),
                "width": self.line_width,
            }
        else:
            # hide
            box_data = {
                "width": 0
            }

        super().set_data(**box_data)


Box = scene.visuals.create_visual_node(BoxVisual)


class TextManager:
    def __init__(self, max_text=100, *args, **kwargs) -> None:
        self.max_text = max_text

        self.free_text_slots = [scene.Text("", *args, **kwargs) for _ in range(max_text)]
        self.assigned_text_slots = {}

    def _set_text(self, text, pos, color):
        if text not in self.assigned_text_slots:
            if len(self.free_text_slots) == 0:
                return None
            
            text_slot = self.free_text_slots.pop()
            text_slot.text = text
            text_slot.color = color
        else:
            text_slot = self.assigned_text_slots.pop(text)

        text_slot.pos = pos

        return text_slot

    def _free_text(self, text):
        text_slot = self.assigned_text_slots.pop(text)
        text_slot.text = ""
        self.free_text_slots.append(text_slot)

    @log_func
    def set_data(self, texts, positions, colors):
        new_text_slots = {}

        for i, (t, p, c) in enumerate(zip(texts, positions, colors)):
            new_text = self._set_text(t, p, c)
            if new_text is not None:
                new_text_slots[t] = new_text

        for t in list(self.assigned_text_slots.keys()):
            self._free_text(t)

        self.assigned_text_slots = new_text_slots


"""
class TextManager:
    def __init__(self, *args, **kwargs) -> None:
        self.text_args = args
        self.text_kwargs = kwargs

        self.texts = []

    def set_data(self, texts, positions):
        for text_vis in self.texts:
            text_vis.parent = None

        self.texts = []

        for (t, p) in zip(texts, positions):
            self.texts.append(scene.Text(t, pos=p, *self.text_args, **self.text_kwargs))
"""


class MOTViewWidget(QWidget):
    def __init__(self, size, show_track_ids=True, parent=None) -> None:
        super().__init__(parent=parent)

        self.w = size[0]
        self.h = size[1]

        self.canvas = scene.SceneCanvas(size=size, resizable=False)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas.native)
        self.setLayout(vbox)

        self.box_lines = Box(size=size, parent=self.canvas.scene)
        self.img_view = ImageView(parent=self.canvas.scene)

        self.tm = TextManager(parent=self.canvas.scene, color="white", anchor_x="left", anchor_y="top") if show_track_ids else None

        self.color_table = colors.as_table()

    def set_data(self, img, boxes):
        self.img_view.set_data(img)

        if boxes is not None and len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * self.w
            boxes[:, [1, 3]] *= self.h

        self.img_view.set_data(img)
        self.box_lines.set_data(boxes)

        if self.tm:
            if boxes is not None and len(boxes) > 0:
                labels = [str(int(b[4])) for b in boxes]
                label_pos = boxes[:, [2, 1]]
                label_colors = self.color_table[boxes[:, 4].astype(int) % len(self.color_table)]
            else:
                labels, label_pos, label_colors = [], [], []

            self.tm.set_data(labels, label_pos, label_colors)
