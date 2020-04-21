from GUI_part.data_preprocessing_ui import Ui_data_preprocessing
from PyQt5.Qt import *
from function_widget import *
import pandas as pd
from numpy import nan


class Data_Preprocessing_Dialog(QDialog, Ui_data_preprocessing):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setupUi(self)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Data_Preprocessing_Dialog()
    window.show()
    sys.exit(app.exec_())
