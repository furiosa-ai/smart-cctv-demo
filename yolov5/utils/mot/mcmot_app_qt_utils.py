
import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QObject, QEvent


class MCMOTQTSignal(QObject):
    mot = pyqtSignal(int, np.ndarray, np.ndarray)  # image_idx, image
    plot = pyqtSignal(str, dict)  # plot_name, lines, line_ids
    fps = pyqtSignal(float)
    finished = pyqtSignal()
    plot_names = None
    parent = None

    def __init__(self) -> None:
        super().__init__()
