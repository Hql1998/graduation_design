from PyQt5.Qt import *
from data_preprocessing_dialog import Data_Preprocessing_Dialog
from function_widget import Function_Widget

class Data_Preprocessing_Function_Widget(Function_Widget):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.class_name = "data_preprocessing"
        self.connected_dialog = None
        self.first_click = True
        print("file_reader object name", self.objectName())
        self.icon_btn.double_clicked.connect(self.icon_btn_double_clicked_handler)

    def icon_btn_double_clicked_handler(self):
        if self.first_click:
            self.connected_dialog = Data_Preprocessing_Dialog(self)
            self.connected_dialog.open()
            self.first_click = False
        else:
            self.connected_dialog.show()

        print("data_preprocessing",self.next_widget, self.previous_widget)