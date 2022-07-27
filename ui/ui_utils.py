import cv2

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui


def convert_img_cv_to_qt(cv_img, size=None):
    """Convert from an opencv image to QPixmap"""
    # rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    rgb_image = cv_img
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    p = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

    if size is not None:
        p = p.scaled(size[0], size[1], Qt.KeepAspectRatio)

    return QPixmap.fromImage(p)
