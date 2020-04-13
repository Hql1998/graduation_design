from PyQt5.Qt import *
from dragable_widget import Dragable_Widget


class MyIconBtn(QPushButton):
    double_clicked = pyqtSignal()

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        e.ignore()

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        e.ignore()

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        e.ignore()

    def mouseDoubleClickEvent(self, e):
        self.double_clicked.emit()



class Function_Widget(Dragable_Widget):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.left_btn_size = QSize(20, 70)
        self.right_btn_size = QSize(20, 60)

        self.setupUi()

    def setupUi(self):

        self.setStyleSheet("""
        Dragable_Widget {
        border:1px solid gray;
        border-radius: 10px;
        background-color: rgba(255,255,255,0);
        }
        Dragable_Widget:hover{
        border-color:white;
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
        self.icon_btn.setStyleSheet("""border:None;background-color:orange;""")
        self.horizontalLayout.addWidget(self.icon_btn)

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

