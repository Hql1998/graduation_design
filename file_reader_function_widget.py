from function_widget import Function_Widget
from file_reader_dialog import File_Reader_Dialog
from PyQt5.Qt import *

class File_Reader_Function_Widget(Function_Widget):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.class_name = "file_reader"
        self.connected_dialog = None
        self.first_click = True
        # self.setObjectName("file_reader_"+ str(len(self.parent().parent().parent().parent().parent().parent().activeWidget["file_reader"])))
        print("file_reader object name", self.objectName())
        self.icon_btn.double_clicked.connect(self.icon_btn_double_clicked_handler)


    def icon_btn_double_clicked_handler(self):
        if self.first_click:
            self.connected_dialog = File_Reader_Dialog(self)
            self.connected_dialog.open()
            self.first_click = False
            # self.connected_dialog.state_changed(self.state_changed_handler)

        else:
            self.connected_dialog.show()








if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = File_Reader_Function_Widget()
    window.show()
    sys.exit(app.exec_())