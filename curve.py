from PyQt5.Qt import *
from curve_label import *
from grid_motion import Grid_Motion

class Curve(QWidget):

    def __init__(self, parent=None, function_widget=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.start_point = QPointF(10, 10)
        self.end_point = QPointF(self.width(), self.height())
        self.function_widget = function_widget
        self.end_label_function_widget = None
        self.setup_ui()
        self.show()

    def setup_ui(self):
        self.resize(20, 20)
        self.setMinimumSize(20, 20)
        self.setStyleSheet("""
        Curve{
        background:None;
        }
        """)
        # border: 1px solid rgba(255,255,0,255);
        # Curve:hover{
        # border: 1px solid black;
        # }
        self.start_label = Start_Label(self)
        self.end_label = End_Label(self)
        self.end_label.move(self.width()-20, self.height()-20)
        self.end_label.moved.connect(self.end_label_moved_handler)
        self.end_label.raise_()

    def paintEvent(self, event):

        super(Curve, self).paintEvent(event)
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        s = self.style()
        s.drawPrimitive(QStyle.PE_Widget, opt, p, self)

        startPoint = self.start_point
        endPoint = self.end_point

        controlPoint1 = QPointF(self.width()/4, self.height()/2 - ((startPoint.y()-endPoint.y())*0.2))
        controlPoint2 = QPointF(self.width()*3/4, self.height()/2 + ((startPoint.y()-endPoint.y())*0.2))

        cubicPath = QPainterPath(startPoint)
        cubicPath.cubicTo(controlPoint1,controlPoint2,endPoint)

        painter = QPainter(self)
        pen = QPen(QColor("gray"), 1)
        painter.setPen(pen)
        painter.drawPath(cubicPath)

    def mousePressEvent(self, e):
        # if self.function_widget is not None:
        #     self.function_widget.raise_()
        self.raise_()
        e.ignore()

    def resizeEvent(self, re):

        self.start_point = self.mapFromGlobal(self.function_widget.right_btn.get_mid_pos())
        self.start_point = QPoint(self.start_point.x()+10, self.start_point.y()+10)
        self.end_point = QPoint((self.width() - self.start_point.x()), abs(self.height() - self.start_point.y()))
        self.start_label.move(self.start_point.x()-10,self.start_point.y()-10)

    def moveEvent(self, e):
        self.old_pos = self.function_widget.right_btn.get_mid_pos()

    def end_label_moved_handler(self, x, y):

        x = x - self.old_pos.x()
        y = y - self.old_pos.y()
        x, y = Grid_Motion.grid_size(x, y)
        old_point_from_draw_area = self.parentWidget().mapFromGlobal(self.old_pos)

        if y <= 0:
            old_point_from_draw_area.setY(old_point_from_draw_area.y() + 20)
            y = y - 40

        rect = QRect(old_point_from_draw_area, QSize(x, y)).normalized()
        self.setGeometry(rect)
        self.end_label.move(self.end_point.x()-10, self.end_point.y()-10)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Curve()
    window.show()
    sys.exit(app.exec_())