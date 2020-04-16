from PyQt5.Qt import *
from function_widget import Function_Widget
import sys

class Draw_Scroll_Area_Content(QScrollArea):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setObjectName("scroll_special")
        self.setup_ui()

    def setup_ui(self):
        self.rb = QRubberBand(QRubberBand.Rectangle,self)

    def mousePressEvent(self, evt):
        self.origin_pos = evt.pos()
        self.rb.setGeometry(QRect(self.origin_pos, QSize()))
        self.rb.setVisible(True)

    def mouseMoveEvent(self, evt):
        size = QSize(evt.pos().x() - self.origin_pos.x(),evt.pos().y() - self.origin_pos.y())
        rect = QRect(self.origin_pos, size).normalized()
        self.rb.setGeometry(rect)

    def mouseReleaseEvent(self, evt):
        draw_scroll_area = self.parent().parent()
        for child in draw_scroll_area.findChildren(Function_Widget):
            if self.rb.geometry().contains(child.geometry()):
                child.icon_btn.toggle()
        self.rb.setVisible(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Draw_Scroll_Area_Content()
    window.show()
    sys.exit(app.exec_())