from GUI_part.file_reader_dialog_ui import Ui_Dialog
from PyQt5.Qt import *


class File_Reader_Dialog(QDialog,Ui_Dialog):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setupUi(self)
        self.open_file_btn.click()
        self.setObjectName("file_reader_dialog")
        self.destroyed.connect(lambda : print("File_Reader_Dialog were destroyed"))

    def open_file_btn_clicked(self):
        print("special", self.open_file_btn.open_result)
        if self.open_file_btn.open_result[0] != "":
            self.display_file_name_label.setText(self.open_file_btn.open_result[0].split("/")[-1])
            self.display_file_name_label.adjustSize()

    def accept(self):
        print("accept")
        self.hide()

    def reject(self):
        print("reject")
        self.hide()




if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = File_Reader_Dialog()
    window.show()
    sys.exit(app.exec_())