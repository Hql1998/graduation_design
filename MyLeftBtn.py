from PyQt5.Qt import *

class MyLeftBtn(QPushButton):

    def __init__(self,parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        # self.setCheckable(True)
        self.setObjectName("left_receiver_btn")

    def mousePressEvent(self, e):
        super().mousePressEvent(e)


    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)