from PyQt5.Qt import *


class Grid_Move:
    grid_width = 20

    @staticmethod
    def get_children_areas_from_parent(widget):
        all_dragable_widgets = widget.parent().findChildren(Dragable_Widget, options=Qt.FindDirectChildrenOnly)
        all_dragable_widgets_rects = [i.geometry() for i in all_dragable_widgets]
        print(all_dragable_widgets_rects)



    @staticmethod
    def grid_move(widget, mouse_e):

        # print(widget.x_direction_dynamic, widget.y_direction_dynamic)
        # Grid_Move.get_children_areas_from_parent(widget)
        # widget_x_forth = widget.x() + int(widget.x_direction_dynamic > 0) * widget.width() + Grid_Move.grid_width * 0.01 * widget.x_direction_dynamic
        # widget_y_forth = widget.y() + int(widget.y_direction_dynamic > 0) * widget.height() + Grid_Move.grid_width * 0.01 * widget.y_direction_dynamic
        #
        # point_x_up = QPoint(widget_x_forth, widget.y())
        # point_x_down = QPoint(widget_x_forth, widget.y() + widget.height())
        #
        # point_y_left = QPoint(widget.x(), widget_y_forth)
        # point_y_right = QPoint(widget.x() + widget.width(), widget_y_forth)

        #print("x_up", widget.parent().childAt(point_x_up),"x_down", widget.parent().childAt(point_x_down))

        vec_x = mouse_e.globalX() - widget.mou_init_x
        vec_y = mouse_e.globalY() - widget.mou_init_y

        des_x = max(0, min(widget.parent().size().width() - 100, widget.win_init_x + vec_x))
        des_y = max(0, min(widget.parent().size().height() - 100, widget.win_init_y + vec_y))

        # if widget.parent().childAt(point_x_up) is not None or widget.parent().childAt(point_x_down) is not None or widget.parent().childAt(point_y_left) is not None or widget.parent().childAt(point_y_right) is not None:
        #     des_x = widget.x()
        #     des_y = widget.y()
        #     # vec_x = mouse_e.globalX() - widget.mouse_x_dynamic
        #     # vec_y = mouse_e.globalY() - widget.mouse_y_dynamic
        #     # pass

        des_x = des_x // Grid_Move.grid_width * Grid_Move.grid_width
        des_y = des_y// Grid_Move.grid_width * Grid_Move.grid_width

        widget.move(des_x, des_y)



class Dragable_Widget(QFrame):
    moveFlag = False

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.moveFlag = True

            self.win_init_x = self.x()
            self.win_init_y = self.y()

            self.mou_init_x = e.globalX()
            self.mou_init_y = e.globalY()

            self.mouse_x_dynamic = e.globalX()
            self.mouse_y_dynamic = e.globalY()

        self.raise_()

    def mouseMoveEvent(self, e):
        if self.moveFlag:

            self.x_direction_dynamic = e.globalX() - self.mouse_x_dynamic
            self.y_direction_dynamic = e.globalY() - self.mouse_y_dynamic

            try:
                self.x_direction_dynamic = self.x_direction_dynamic / abs(self.x_direction_dynamic)
            except:
                self.x_direction_dynamic = 0
            try:
                self.y_direction_dynamic = self.y_direction_dynamic / abs(self.y_direction_dynamic)
            except:
                self.y_direction_dynamic = 0

            self.mouse_x_dynamic = e.globalX()
            self.mouse_y_dynamic = e.globalY()

            Grid_Move.grid_move(self, e)


    def mouseReleaseEvent(self, e):
        self.moveFlag = False
