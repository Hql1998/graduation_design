from PyQt5.Qt import *


class Start_Label(QLabel):
    def __init__(self,parent = None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.not_find = True
        self.setup_ui()

    def setup_ui(self):
        self.resize(20, 20)
        self.setPixmap(QPixmap(":/main_window/right_arrow.png").scaled(20, 20))
        self.setStyleSheet("background:red;")

class End_Label(QLabel):

    moved = pyqtSignal(int, int)
    pressed = pyqtSignal()

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setup_ui()
        self.draw_area_content = self.parentWidget().parentWidget()

    def setup_ui(self):
        self.resize(20, 20)
        self.setPixmap(QPixmap(":/main_window/right_arrow.png").scaled(20, 20))
        self.setStyleSheet("background:green;")

    def mousePressEvent(self, me):
        # self.mouse_start = self.parentWidget().start_label.mapToGlobal(QPoint(0,0))
        self.pressed.emit()

    def mouseMoveEvent(self, me):

        distance = me.globalPos()
        vec_x = int(distance.x())
        vec_y = int(distance.y()) + 20

        self.moved.emit(vec_x, vec_y)

    def mouseReleaseEvent(self, me):
        self.draw_area_content.get_function_widget()
        self.not_find = True
        for function_widget in self.draw_area_content.function_widget_set:
            if function_widget is not self.parentWidget().function_widget and function_widget.class_name in self.parentWidget().function_widget.allowed_next_fun_widget_list:
                if function_widget.geometry().contains(self.draw_area_content.mapFromGlobal(me.globalPos())):
                    if function_widget.present_previouse_num < function_widget.allowed_previouse_num and self.parentWidget().function_widget.present_next_num < self.parentWidget().function_widget.allowed_next_num:
                        if self.parentWidget().function_widget not in function_widget.previous_widgets:
                            function_widget.present_previouse_num += 1
                            self.parentWidget().function_widget.present_next_num += 1
                            self.parentWidget().function_widget.right_btn.start_curve()
                            self.not_find = False
                            left_sucket_point = function_widget.get_left_sucket()
                            function_widget.previous_widgets.append(self.parentWidget().function_widget)
                            self.parentWidget().end_label_function_widget = function_widget
                            function_widget.update_data_from_previous()
                            self.parentWidget().function_widget.next_widgets.append(function_widget)
                            self.parentWidget().end_label_moved_handler(left_sucket_point.x()+20, left_sucket_point.y()+20)

        if self.not_find:
            self.move(0, 0)
            self.parentWidget().move(self.parentWidget().parentWidget().mapFromGlobal(self.parentWidget().function_widget.right_btn.get_mid_pos()))
            self.parentWidget().end_point = QPoint(0, 0)
            self.parentWidget().start_point = QPoint(0, 0)
            self.parentWidget().resize(20, 20)

            if self.parentWidget().function_widget is not None:
                if self.parentWidget().function_widget.next_widgets != []:
                    if self.parentWidget().end_label_function_widget is not None:
                        connect_widget_index = self.parentWidget().function_widget.next_widgets.index(self.parentWidget().end_label_function_widget)
                        del self.parentWidget().function_widget.next_widgets[connect_widget_index]
                        connect_widget_index = self.parentWidget().end_label_function_widget.previous_widgets.index(self.parentWidget().function_widget)
                        del self.parentWidget().end_label_function_widget.previous_widgets[connect_widget_index]
                        self.parentWidget().function_widget.present_next_num -= 1
                        self.parentWidget().end_label_function_widget.present_previouse_num -= 1
            self.parentWidget().end_label_function_widget = None
            self.parentWidget().raise_()