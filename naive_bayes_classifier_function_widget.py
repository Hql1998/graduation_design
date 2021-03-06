from PyQt5.Qt import *
from naive_bayes_classifier_dialog import Naive_Bayes_Classifier_Dialog
from function_widget import Function_Widget
import copy

class Naive_Bayes_Classifier_Function_Widget(Function_Widget):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.class_name = "naive_bayes_classifier"
        self.connected_dialog = None
        self.first_click = True
        self.allowed_next_fun_widget_list = []
        self.allowed_previouse_num = 1
        self.allowed_next_num = 1
        self.data = None
        print("Naive Bayes object name", self.objectName())
        self.icon_btn.double_clicked.connect(self.icon_btn_double_clicked_handler)


    def icon_btn_double_clicked_handler(self):

        if self.previous_widgets != [] and self.data is not None:
            if self.first_click:
                self.connected_dialog = Naive_Bayes_Classifier_Dialog(self)
                self.update_data_from_previous()
                self.connected_dialog.open()
                self.connected_dialog.reset_btn.clicked.connect(self.update_data_from_previous)
                self.first_click = False
            else:
                self.connected_dialog.show()

        print("Naive Bayes Classifier ", self.next_widgets, self.previous_widgets)

    def update_data_from_previous(self):
        if self.previous_widgets != [] and self.previous_widgets[0].data is not None:
            if self.data is None:
                self.data = copy.deepcopy(self.previous_widgets[0].data)
                if self.connected_dialog is not None:
                    self.connected_dialog.data = self.data
                    self.connected_dialog.get_y_labels()
            else:
                QErrorMessage.qtHandler()
                QErrorMessage
                qInfo("previous"+str(self.previous_widgets[0].data["train_x"].shape)+str(self.class_name)+str(self.data["train_x"].shape))
                self.data = copy.deepcopy(self.previous_widgets[0].data)
                self.connected_dialog.data = self.data
                self.connected_dialog.get_y_labels()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Naive_Bayes_Classifier_Function_Widget()
    window.show()
    sys.exit(app.exec_())