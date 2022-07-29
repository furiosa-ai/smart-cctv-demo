

from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QStyle, QSlider, QFileDialog
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread


class _SelectFilePromptWidget(QWidget):
    file_selected = pyqtSignal(str)

    def __init__(self, title, parent=None) -> None:
        super().__init__(parent)

        lay = QVBoxLayout()

        open_file_button = QPushButton("Open file")
        open_file_button.clicked.connect(self.open_file_dialog)

        lay.addWidget(QLabel(title))
        lay.addWidget(open_file_button)
        lay.setAlignment(Qt.AlignTop)

        self.setLayout(lay)

    def open_file_dialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')

        if not fname[0]:
            return
        else:
            fname = fname[0]
        
        self.file_selected.emit(fname)


class _FileDisplay(QWidget):
    close_clicked = pyqtSignal()

    def __init__(self, widget, parent=None) -> None:
        super().__init__(parent)

        self.widget = widget

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close_clicked)

        lay = QVBoxLayout()
        lay.addWidget(widget)
        lay.addWidget(close_button)
        self.setLayout(lay)


class SelectFileWidget(QWidget):
    file_selected = pyqtSignal(str)
    file_closed = pyqtSignal()

    def __init__(self, title, cont_widget_builder, file=None, parent=None) -> None:
        super().__init__(parent)

        self.cont_widget_builder = cont_widget_builder

        self.lay = QVBoxLayout()
        self.setLayout(self.lay)

        self.file = None

        self.file_select = _SelectFilePromptWidget(title)
        self.file_select.file_selected.connect(self.open_file)

        self.cont_widget = None

        self.update_state(file=file)

    def open_file_dialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')

        if not fname[0]:
            return
        else:
            fname = fname[0]
        
        self.open_file(fname)

    def open_file(self, file):
        self.update_state(file=file)
        self.file_selected.emit(file)

    def close_file(self):
        self.update_state(file=None)
        self.file_selected.emit(None)
        self.file_closed.emit()

    def get_widget(self):
        return self.cont_widget.widget

    def update_state(self, **state):
        for k, v in state.items():
            assert hasattr(self, k)
            setattr(self, k, v)

        if self.file is None:
            if self.cont_widget is not None:
                self.cont_widget.setParent(None)
                self.cont_widget = None

            self.lay.addWidget(self.file_select)
        else:
            self.lay.removeWidget(self.file_select)
            self.file_select.setParent(None)
            self.cont_widget = _FileDisplay(self.cont_widget_builder(self.file))
            self.cont_widget.close_clicked.connect(self.close_file)
            self.lay.addWidget(self.cont_widget)
