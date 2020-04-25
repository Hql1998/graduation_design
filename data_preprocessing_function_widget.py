from PyQt5.Qt import *
from data_preprocessing_dialog import Data_Preprocessing_Dialog
from function_widget import Function_Widget

class Data_Preprocessing_Function_Widget(Function_Widget):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.class_name = "data_preprocessing"
        self.connected_dialog = None
        self.first_click = True
        self.allowed_next_fun_widget_list = []
        self.dataFrame = None
        print("data_preprocssing object name", self.objectName())
        self.icon_btn.double_clicked.connect(self.icon_btn_double_clicked_handler)
        self.destroyed.connect(lambda obj: qApp.main_window.log_te.append("\n" + str(obj) + "Function_Widget deleted"))


    def icon_btn_double_clicked_handler(self):

        if self.previous_widgets != [] and self.dataFrame is not None:
            if self.first_click:
                self.connected_dialog = Data_Preprocessing_Dialog(self)
                self.connected_dialog.open()
                self.connected_dialog.reset_btn.clicked.connect(self.update_data_from_previous)
                self.first_click = False
            else:
                self.connected_dialog.show()

        print("data_preprocessing", self.next_widgets, self.previous_widgets)


    def update_data_from_previous(self):
        if self.previous_widgets != [] and self.previous_widgets[0].dataFrame is not None:
            if self.previous_widgets[0].dataFrame is not self.dataFrame:
                self.dataFrame = self.previous_widgets[0].dataFrame
                if self.connected_dialog is not None:
                    self.connected_dialog.update_data()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Data_Preprocessing_Function_Widget()
    window.show()
    sys.exit(app.exec_())