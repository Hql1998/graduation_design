from PyQt5.Qt import *
from global_variables import *
import sys, os

# #mid_size_widget, default_path_file_path# imports from ./global_variables.py

class Open_Model_Btn(QPushButton):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.open_result = ("", "")
        self.clicked.connect(self.open_file)

    # @staticmethod
    def open_file(self):

        if not os.path.isdir("./temp"):
            os.mkdir("./temp")
        if not os.path.isfile(default_path_file_path):
            default_path_file = open(default_path_file_path, "w", encoding="UTF - 8")
            default_path_str = "../"
        else:
            default_path_file = open(default_path_file_path, 'r+', encoding="UTF - 8")
            default_path_str = default_path_file.read()
            if len(default_path_str) < 1:
                default_path_str = "../"

        open_re = QFileDialog.getOpenFileName(self, "open a model", default_path_str,
                                              "sklearn joblib(*.joblib)", "sklearn joblib(*.joblib)")

        file_path = open_re[0]
        if len(file_path) > 0:
            dir_list = file_path.split("/")[0:-1]
            default_path_str = "/".join(dir_list) + "/"
            default_path_file.seek(0, 0)
            default_path_file.truncate()
            default_path_file.write(default_path_str)
            default_path_file.close()
            if self.open_result[0] != open_re[0]:
                self.open_result = open_re
        else:
            default_path_file.close()

