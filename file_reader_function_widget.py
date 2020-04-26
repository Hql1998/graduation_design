from function_widget import Function_Widget
from file_reader_dialog import File_Reader_Dialog
from PyQt5.Qt import *

class File_Reader_Function_Widget(Function_Widget):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.class_name = "file_reader"
        self.connected_dialog = None
        self.first_click = True
        self.data = None
        self.allowed_next_fun_widget_list = ["deal_empty", "data_preprocessing"]
        print("file_reader object name", self.objectName())
        self.icon_btn.double_clicked.connect(self.icon_btn_double_clicked_handler)


    def icon_btn_double_clicked_handler(self):
        if self.first_click:
            self.connected_dialog = File_Reader_Dialog(self)
            self.connected_dialog.open()
            self.first_click = False
        else:
            self.connected_dialog.show()

        print("file_reader", self.next_widgets, self.previous_widgets)








if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = File_Reader_Function_Widget()
    window.show()
    sys.exit(app.exec_())