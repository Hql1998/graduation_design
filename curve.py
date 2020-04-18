from PyQt5.Qt import *
from grid_motion import Grid_Motion

class Start_Label(QLabel):
    def __init__(self,parent = None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setup_ui()

    def setup_ui(self):
        self.resize(20, 20)
        self.setPixmap(QPixmap(":/main_window/right_arrow.png").scaled(20, 20))
        self.setStyleSheet("background:None;")

class End_Label(QLabel):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setup_ui()

    def setup_ui(self):
        self.resize(20, 20)
        self.setPixmap(QPixmap(":/main_window/right_arrow.png").scaled(20, 20))
        self.setStyleSheet("background:None;")

    def mousePressEvent(self, me):
        self.mouse_start = self.parentWidget().start_label.mapToGlobal(QPoint(0,0))

    def mouseMoveEvent(self, me):
        size = QSize(me.screenPos().x() - self.mouse_start.x() + 20, me.screenPos().y() - self.mouse_start.y() + 20)
        size = Grid_Motion.grid_size(size)
        self.parentWidget().resize(size)

    def mouseReleaseEvent(self, me):
        pass



class Curve(QWidget):

    def __init__(self, parent=None,*args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.start_point = QPointF(10, 10)
        self.end_point = QPointF(self.width(), self.height())
        self.negative = False
        self.setup_ui()
        self.show()

    def setup_ui(self):
        self.resize(20, 20)
        self.setMinimumSize(20, 20)
        self.setStyleSheet("""
        Curve{
        border: 1px solid rgba(255,255,0,255);
        background:None;
        }
        Curve:hover{
        border: 1px solid black;
        }
        """)

        self.start_label = Start_Label(self)
        self.end_label = End_Label(self)
        self.end_label.move(self.width()-20, self.height()-20)
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
        painter.begin(self)
        painter.drawPath(cubicPath)
        painter.end()

    def resizeEvent(self, re):
        self.end_point = QPointF(re.size().width()-10, re.size().height()-10)
        self.end_label.move(re.size().width() - 20, re.size().height() - 20)




if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Curve()
    window.show()
    sys.exit(app.exec_())