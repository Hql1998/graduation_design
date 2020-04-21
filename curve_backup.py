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

    sign_changed = pyqtSignal()

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setup_ui()
        self.draw_area_content = self.parentWidget().parentWidget()
        self.old_size = QSize()

    def setup_ui(self):
        self.resize(20, 20)
        self.setPixmap(QPixmap(":/main_window/right_arrow.png").scaled(20, 20))
        self.setStyleSheet("background:None;")

    def mousePressEvent(self, me):
        self.mouse_start = self.parentWidget().start_label.mapToGlobal(QPoint(0,0))

    def mouseMoveEvent(self, me):
        size=QSize(me.screenPos().x()-self.mouse_start.x()+20, me.screenPos().y()-self.mouse_start.y()+20)
        size = Grid_Motion.grid_size(size)
        if size.height() < 0:
            if self.parentWidget().negative == False:
                pass
            self.parentWidget().negative = True
            # self.parentWidget().move(self.parentWidget().x(), self.parentWidget().y() + size.height() - self.old_size.height())
            self.parentWidget().resize(size.width(), -size.height()+20)
        elif size.height() >= 20:
            if self.parentWidget().negative == True and self.parentWidget().height() > 20:

                self.sign_changed.emit()

            self.parentWidget().negative = False
            self.parentWidget().resize(size)


        self.old_size = size

    def mouseReleaseEvent(self, me):
        self.draw_area_content.get_function_widget()
        self.not_find = True
        for function_widget in self.draw_area_content.function_widget_set:
            if function_widget is not self.parentWidget().function_widget:
                if function_widget.geometry().contains(self.draw_area_content.mapFromGlobal(me.globalPos())):
                    self.not_find = False
                    left_sucket_point = function_widget.get_left_sucket()
                    function_widget.previous_widget = self.parentWidget().function_widget
                    self.parentWidget().function_widget.next_widget = function_widget
                    self.parentWidget().resize_by_left_sucket_point(left_sucket_point)

        if self.not_find:
            self.parentWidget().resize(20, 20)
            if self.parentWidget().negative:
                self.parentWidget().move(self.parentWidget().function_widget.right_btn.get_mid_pos())
            if self.parentWidget().function_widget is not None:
                if self.parentWidget().function_widget.next_widget is not None:
                    self.parentWidget().function_widget.next_widget.previous_widget = None
                    self.parentWidget().function_widget.next_widget = None

        self.raise_()




class Curve(QWidget):

    def __init__(self, parent=None,*args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.start_point = QPointF(10, 10)
        self.end_point = QPointF(self.width(), self.height())
        self.negative = False
        self.function_widget = None
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
        self.end_label.sign_changed.connect(self.handle_sign_changed)
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

    def mousePressEvent(self, e):
        if self.function_widget is not None:
            self.function_widget.raise_()

    def resizeEvent(self, re):
        # # self.end_point = QPointF(re.size().width()-10, re.size().height()-10)
        # self.end_label.move(re.size().width() - 20, re.size().height() - 20)

        if self.negative:
            self.move(self.x(), self.y() - re.size().height() - self.end_label.old_size.height() + 20)
            self.start_point = QPoint(10, re.size().height() - 10)
            self.end_point = QPoint(re.size().width() - 10, 10)

            self.end_label.move(re.size().width() - 20, 0)
            self.start_label.move(0, re.size().height()-20)
        else:
            self.start_point = QPoint(10, 10)
            self.end_point = QPointF(re.size().width() - 10, re.size().height() - 10)

            self.start_label.move(0, 0)
            self.end_label.move(re.size().width() - 20, re.size().height() - 20)

    def resize_by_left_sucket_point(self, left_sucket_point):

        size = QSize(left_sucket_point.x() - self.start_label.mapToGlobal(QPoint(0,0)).x() + 20,
                     left_sucket_point.y() - self.start_label.mapToGlobal(QPoint(0,0)).y() + 20)

        if size.height() < 0:
            self.negative = True
            self.resize(size.width(), -size.height() + 20)
        else:
            self.negative = False
            self.resize(size)
        self.end_label.old_size = size

    def handle_sign_changed(self):
        self.move(self.x(), self.y() + 20)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Curve()
    window.show()
    sys.exit(app.exec_())