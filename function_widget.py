from dragable_widget import Dragable_Widget
from MyIconBtn import MyIconBtn
from MyLeftBtn import MyLeftBtn
from MyRightBtn import MyRightBtn
from light_label import *




class Function_Widget(Dragable_Widget):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.left_btn_size = QSize(20, 70)
        self.right_btn_size = QSize(20, 60)

        self.class_name = "function_widget"
        self.next_widget = None
        self.previous_widget = None

        self.setupUi()

    def setupUi(self):
        self.setStyleSheet("""
        Dragable_Widget {
        border:1px solid gray;
        border-radius: 10px;
        background: rgba(255,255,255,0);

        }
        Dragable_Widget:hover{
        border-color:white;
        }
        MyIconBtn:checked{
        border:2px solid rgb(0,255,255);
        }
        MyIconBtn{
        border:None;
        background: orange;
        }
        MyRightBtn{
        border:None;
        background: red
        }
        MyLeftBtn{
        border:None;
        background:orange;
        }
        """)
        self.setFixedSize(QSize(100, 100))

        self.horizontalLayout = QHBoxLayout(self)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)

        self.left_btn = MyLeftBtn(self)
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

        self.right_btn = MyRightBtn(self)
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
        background: rgba(255,0,0,50);
        color:white;}
        QPushButton:hover {
        background: rgba(255,0,0,220);}
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

        self.right_btn.curve.deleteLater()
        if self.previous_widget is not None:
            self.previous_widget.right_btn.curve.end_label_moved_handler(self.previous_widget.right_btn.get_mid_pos().x()+20,
                                                                         self.previous_widget.right_btn.get_mid_pos().y()+20)
            self.previous_widget.next_widget = None
        if self.next_widget is not None:
            self.next_widget.previous_widget = None
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

    def paintEvent(self, event):
        super(Dragable_Widget, self).paintEvent(event)
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        s = self.style()
        s.drawPrimitive(QStyle.PE_Widget, opt, p, self)

    def moveEvent(self, me):
        point = self.get_left_sucket()
        if self.right_btn.curve is not None:
            self.right_btn.curve.move(me.pos()+QPoint(8/10*self.width(), 1/2*self.height()-10))
        if self.previous_widget is not None:
            self.previous_widget.right_btn.curve.end_label_moved_handler(point.x()+20, point.y()+20)
        if self.next_widget is not None:
            self.right_btn.curve.end_label_moved_handler(self.next_widget.get_left_sucket().x() +20, self.next_widget.get_left_sucket().y()+20)

    def get_left_sucket(self):
        return self.mapToGlobal(QPoint(0,0))+QPoint(0*self.width(), 1/2*self.height() - 10)
