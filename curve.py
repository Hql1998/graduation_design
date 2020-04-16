from PyQt5.Qt import *


class Curve(QWidget):

    def __init__(self):
        super().__init__()
        self.start_point = QPointF(0, 0)
        self.end_point = QPointF(self.width(), self.height())

    def paintEvent(self, event):

        startPoint = self.start_point
        endPoint = self.end_point

        controlPoint1 = QPointF(self.width()/4, self.height()/2 - ((startPoint.y()-endPoint.y())*0.2))
        controlPoint2 = QPointF(self.width()*3/4, self.height()/2 + ((startPoint.y()-endPoint.y())*0.2))

        cubicPath = QPainterPath(startPoint)
        cubicPath.cubicTo(controlPoint1,controlPoint2,endPoint)

        painter = QPainter(self)
        pen = QPen(QColor("gray"), 2)
        painter.setPen(pen)
        painter.begin(self)
        painter.drawPath(cubicPath)
        painter.end()

import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Curve()
    window.show()
    sys.exit(app.exec_())