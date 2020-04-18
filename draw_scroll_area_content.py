from PyQt5.Qt import *
from function_widget import Function_Widget
import sys

class Draw_Scroll_Area_Content(QScrollArea):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setObjectName("scroll_special")
        self.function_widget_set = []
        self.setup_ui()


    def setup_ui(self):
        self.rb = QRubberBand(QRubberBand.Rectangle, self)
        self.rb.setObjectName("rubberband")


    def mousePressEvent(self, evt):
        self.get_function_widget()
        self.origin_pos = evt.pos()
        self.rb.setGeometry(QRect(self.origin_pos, QSize()))
        self.rb.setVisible(True)

        if len(self.function_widget_set) > 0:
            pass

    def mouseMoveEvent(self, evt):
        size = QSize(evt.pos().x() - self.origin_pos.x(), evt.pos().y() - self.origin_pos.y())
        rect = QRect(self.origin_pos, size).normalized()
        self.rb.setGeometry(rect)

    def mouseReleaseEvent(self, evt):
        for child in self.function_widget_set:
            if self.rb.geometry().contains(child.geometry()):
                child.icon_btn.toggle()
        self.rb.setVisible(False)

    def get_function_widget(self):
        self.function_widget_set = self.parent().findChildren(Function_Widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Draw_Scroll_Area_Content()
    window.show()
    sys.exit(app.exec_())