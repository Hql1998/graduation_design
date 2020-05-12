from PyQt5.Qt import *
from deal_with_empty_value_dialog import Deal_With_Empty_Dialog
from function_widget import Function_Widget
import copy

class Deal_With_Empty_Value_Function_Widget(Function_Widget):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.class_name = "deal_empty"
        self.connected_dialog = None
        self.first_click = True
        self.allowed_next_fun_widget_list = ["data_preprocessing", "random_forest_classifier"]
        self.data = None
        print("deal with empty object name", self.objectName())
        self.icon_btn.double_clicked.connect(self.icon_btn_double_clicked_handler)
        self.destroyed.connect(lambda obj: qApp.main_window.log_te.append("\n" + str(obj) + "Function_Widget deleted"))


    def icon_btn_double_clicked_handler(self):

        if self.previous_widgets != [] and self.data is not None:
            if self.first_click:
                self.connected_dialog = Deal_With_Empty_Dialog(self)
                self.connected_dialog.open()
                self.connected_dialog.reset_btn.clicked.connect(self.update_data_from_previous)
                self.first_click = False
            else:
                self.connected_dialog.show()

        print("deal with empty", self.next_widgets, self.previous_widgets)

    def update_data_from_previous(self):
        if self.previous_widgets != [] and self.previous_widgets[0].data is not None:
            if self.data is None:
                self.data = copy.deepcopy(self.previous_widgets[0].data)
                if self.connected_dialog is not None:
                    self.connected_dialog.update_data()
            else:
                self.data = copy.deepcopy(self.previous_widgets[0].data)
                self.connected_dialog.data = self.data
                if self.connected_dialog is not None:
                    self.connected_dialog.update_data()