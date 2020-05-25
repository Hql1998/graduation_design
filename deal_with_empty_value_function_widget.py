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
        self.allowed_next_fun_widget_list = ["data_preprocessing", "lasso_logistic_regression", "svm_classifier", "random_forest_classifier", "naive_bayes_classifier", "knn_classifier"]
        self.data = None
        self.allowed_previouse_num = 1
        self.allowed_next_num = 10
        print("deal with empty object name", self.objectName())
        self.icon_btn.double_clicked.connect(self.icon_btn_double_clicked_handler)


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