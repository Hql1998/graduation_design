from PyQt5.Qt import *
from dragable_widget import Dragable_Widget


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




class Lights_Change:
    @staticmethod
    def unprepared(light_label):
        light_label.setStyleSheet("""
        border-radius:9px;
        background-color: gray;
        """)
        light_label.status = "unprepared"
    @staticmethod
    def start(light_label):
        light_label.setStyleSheet("""
        border-radius:9px;
        background-color: rgb(85,217,132);
        """)
        light_label.status = "start"
    @staticmethod
    def processing(light_label):
        light_label.setStyleSheet("""
        border-radius:9px;
        background-color: yellow;
        """)
        light_label.status = "processing"
    @staticmethod
    def finish(light_label):
        light_label.setStyleSheet("""
        border-radius:9px;
        background-color: red;
        """)
        light_label.status = "finish"


class Light_Label(QLabel):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.status = "unprepared"
        self.setup_ui()

    def setup_ui(self):
        self.setFixedSize(18,18)
        self.setText("")
        Lights_Change.unprepared(self)




class Function_Widget(Dragable_Widget):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.left_btn_size = QSize(20, 70)
        self.right_btn_size = QSize(20, 60)

        self.class_name = "function_widget"
        self.next_widget = ""

        self.setupUi()

    def setupUi(self):

        self.setStyleSheet("""
        Dragable_Widget {
        border:1px solid gray;
        border-radius: 10px;
        background-color: rgba(255,255,255,0);
        background-image:None;
        }
        Dragable_Widget:hover{
        border-color:white;
        }
        MyIconBtn:checked{
        border:2px solid rgb(0,255,255);
        }
        MyIconBtn{
        border:None;
        background-color:orange;
        }
        """)
        self.setFixedSize(QSize(100, 100))

        self.horizontalLayout = QHBoxLayout(self)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)

        self.left_btn = QPushButton(self)
        self.left_btn.setFixedSize(self.left_btn_size)
        self.left_btn.setFlat(True)
        self.left_btn.setObjectName("left_btn")
        self.left_btn.setText("<")
        self.left_btn.setStyleSheet("border:None;background-color: rgba(255,255,255,0);")
        self.horizontalLayout.addWidget(self.left_btn)

        self.icon_btn = MyIconBtn(self)
        self.icon_btn.setFixedSize(QSize(60, 60))
        self.icon_btn.setFlat(False)
        self.icon_btn.setObjectName("icon_btn")
        self.icon_btn.setText("icon")


        self.light_label = Light_Label(parent=self)
        self.mid_vertical_layout = QVBoxLayout(self)
        self.light_h_layout = QHBoxLayout(self)
        self.light_h_layout.addWidget(self.light_label)

        self.mid_vertical_layout.addStretch(1)
        self.mid_vertical_layout.addWidget(self.icon_btn, 3)
        self.mid_vertical_layout.addLayout(self.light_h_layout, 1)

        self.horizontalLayout.addLayout(self.mid_vertical_layout)

        self.right_btn = QPushButton(self)
        self.right_btn.setFixedSize(self.right_btn_size)
        self.right_btn.setFlat(True)
        self.right_btn.setObjectName("right_btn")
        self.right_btn.setText(">")
        self.right_btn.setStyleSheet("border:None;background-color: rgba(255,255,255,0);")

        self.close_btn = QPushButton(self)
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setFlat(True)
        self.close_btn.setObjectName("close_btn")
        self.close_btn.setText("X")
        self.close_btn.setStyleSheet("""
        QPushButton {
        border-radius:10px;
        background-color: rgba(255,0,0,50);
        color:white;}
        QPushButton:hover {
        background-color: rgba(255,0,0,220);}
        """)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.addWidget(self.close_btn)
        self.verticalLayout.addWidget(self.right_btn)
        self.verticalLayout.addStretch(1)
        self.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(0, 3)

        self.horizontalLayout.addLayout(self.verticalLayout)

        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 6)
        self.horizontalLayout.setStretch(2, 2)

        self.close_btn.clicked.connect(self.close_btn_handler)

        return None

    def close_btn_handler(self):
        self.deleteLater()

    def state_changed_handler(self, state):
        if state == "unprepared":
            Lights_Change.unprepared(self.light_label)
        elif state == "start":
            Lights_Change.start(self.light_label)
        elif state == "processing":
            Lights_Change.processing(self.light_label)
        elif state == "finish":
            Lights_Change.finish(self.light_label)

