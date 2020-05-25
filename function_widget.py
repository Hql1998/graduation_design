from dragable_widget import Dragable_Widget
from MyIconBtn import MyIconBtn
from MyLeftBtn import MyLeftBtn
from MyRightBtn import MyRightBtn
from light_label import *
from print_to_log import *




class Function_Widget(Dragable_Widget):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.left_btn_size = QSize(20, 70)
        self.right_btn_size = QSize(20, 60)

        self.class_name = "function_widget"
        self.next_widgets = []
        self.previous_widgets = []
        self.allowed_next_fun_widget_list = []
        self.allowed_previouse_num = 1
        self.allowed_next_num = 1
        self.present_previouse_num = 0
        self.present_next_num = 0

        self.data = None
        self.connected_dialog = None

        self.setupUi()
        self.destroyed.connect(lambda x: print_to_log(x, self.class_name, "were destroyed"))

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
        self.mid_vertical_layout = QVBoxLayout()
        self.light_h_layout = QHBoxLayout()
        self.light_h_layout.addWidget(self.light_label)

        self.mid_vertical_layout.addStretch(1)
        self.mid_vertical_layout.addWidget(self.icon_btn, 3)
        self.mid_vertical_layout.addLayout(self.light_h_layout, 1)

        self.horizontalLayout.addLayout(self.mid_vertical_layout)

        self.right_btn = MyRightBtn(self)
        self.right_btn.start_curve()
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
        for curve in self.right_btn.curves:
            curve.deleteLater()
        if self.previous_widgets != []:
            for previous_widget in self.previous_widgets:
                for curve in previous_widget.right_btn.curves:
                    if curve.end_label_function_widget is self:
                        curve.end_label_moved_handler(previous_widget.right_btn.get_mid_pos().x()+20, previous_widget.right_btn.get_mid_pos().y()+20)
                        curve.end_label_function_widget = None
                        curve.raise_()
                        previous_widget.present_next_num -= 1
                try:
                    del previous_widget.next_widgets[previous_widget.next_widgets.index(self)]
                except ValueError as e:
                    print(e)

        if self.next_widgets != []:
            for next_widget in self.next_widgets:
                try:
                    del next_widget.previous_widgets[next_widget.previous_widgets.index(self)]
                except ValueError as e:
                    print(e)
        self.deleteLater()
        qApp.main_window.function_widget_dict[self.class_name].remove(self)
        print(qApp.main_window.function_widget_dict)


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
        if self.right_btn.curves != []:
            for curve in self.right_btn.curves:
                curve.move(me.pos()+QPoint(8/10*self.width(), 1/2*self.height()-10))
        if self.previous_widgets != []:
            for previous_widget in self.previous_widgets:
                for curve in previous_widget.right_btn.curves:
                    if curve.end_label_function_widget is self:
                        curve.end_label_moved_handler(point.x()+20, point.y()+20)
        if self.next_widgets != []:
            for next_widget in self.next_widgets:
                for curve in self.right_btn.curves:
                    if curve.end_label_function_widget is next_widget:
                        curve.end_label_moved_handler(next_widget.get_left_sucket().x() +20, next_widget.get_left_sucket().y()+20)

    def get_left_sucket(self):
        return self.mapToGlobal(QPoint(0,0))+QPoint(0*self.width(), 1/2*self.height() - 10)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Function_Widget()
    window.show()
    sys.exit(app.exec_())
