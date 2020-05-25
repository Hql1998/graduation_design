from PyQt5.Qt import *
from curve import Curve


class MyRightBtn(QPushButton):

    def __init__(self,parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        # self.setCheckable(True)
        self.setObjectName("right_emiter_btn")
        self.curves = []


    def start_curve(self):

        if self.parentWidget().present_next_num < self.parentWidget().allowed_next_num:
            start_point = self.get_mid_pos()
            curve = Curve(self.parentWidget().parentWidget(), function_widget=self.parentWidget())
            curve.move(self.parentWidget().parentWidget().mapFromGlobal(start_point))
            curve.raise_()
            self.curves.append(curve)


    def get_mid_pos(self):
         return self.parentWidget().mapToGlobal(QPoint(0,0)) + QPoint(4 / 5 * self.parentWidget().width(),
                                           1 / 2 * self.parentWidget().height() - 10)
