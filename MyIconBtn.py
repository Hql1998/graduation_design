from PyQt5.Qt import *

class MyIconBtn(QPushButton):
    double_clicked = pyqtSignal()

    def __init__(self,parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setCheckable(True)
        # self.setChecked(True)
        self.setObjectName("iconbtn")

    def mousePressEvent(self, e):
        # super().mousePressEvent(e)
        e.ignore()

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        e.ignore()

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        e.ignore()

    def mouseDoubleClickEvent(self, e):
        self.double_clicked.emit()