from PyQt5.Qt import *
from curve import Curve


class MyRightBtn(QPushButton):

    def __init__(self,parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        # self.setCheckable(True)
        self.setObjectName("right_emiter_btn")
        self.receiver_btn = None
        self.curve = None
        self.start_curve()

    def start_curve(self):
        if self.curve is None:
            start_point = self.get_mid_pos()
            self.curve = Curve(self.parentWidget().parentWidget())
            self.curve.move(start_point)
        else:
            self.curve.end_label.raise_()

    def get_mid_pos(self):
         return self.parentWidget().pos() + QPoint(4 / 5 * self.parentWidget().width(),
                                           1 / 2 * self.parentWidget().height() - 10)
