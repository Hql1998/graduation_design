from PyQt5.Qt import *
from grid_motion import Grid_Motion


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
        if self.right_btn.curve is not None:
            self.right_btn.curve.raise_()


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
            Grid_Motion.grid_move(self, e)


    def mouseReleaseEvent(self, e):
        self.moveFlag = False

    def moveEvent(self, me):
        pass

