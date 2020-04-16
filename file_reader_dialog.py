from GUI_part.file_reader_dialog_ui import Ui_file_reader_dialog
from PyQt5.Qt import *
from function_widget import *
import pandas as pd
from numpy import nan

class File_Reader_Dialog(QDialog,Ui_file_reader_dialog):

    state_changed = pyqtSignal(str)

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setupUi(self)
        # self.open_file_btn.click()
        self.setObjectName("file_reader_dialog")
        self.destroyed.connect(lambda: print("File_Reader_Dialog were destroyed"))


    def open_file_btn_clicked(self):
        if self.open_file_btn.open_result[0] != "":
            self.parentWidget().state_changed_handler("start")

            self.display_file_name_label.setText(self.open_file_btn.open_result[0].split("/")[-1])
            self.display_file_name_label.adjustSize()
            self.display_table()
            self.display_table_describe()





    def display_table(self):

        self.parentWidget().state_changed_handler("processing")
        if self.open_file_btn.open_result[1] == "excel(*.xlsx)":
            self.dataFrame = pd.read_excel(self.open_file_btn.open_result[0])
        elif self.open_file_btn.open_result[1] == "csv(*.csv)":
            self.dataFrame = pd.read_csv(self.open_file_btn.open_result[0])
        elif self.open_file_btn.open_result[1] == "tsv(*.tsv)":
            self.dataFrame = pd.read_csv(self.open_file_btn.open_result[0], sep="\t")


        data = self.dataFrame.replace(nan, 'N/A')
        header_data = data.columns.to_list()

        self.tableWidget.setRowCount(data.shape[0])
        self.tableWidget.setColumnCount(data.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(header_data)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = QTableWidgetItem(str(data.iloc[i, j]))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if data.iloc[i, j] == 'N/A':
                    item.setBackground(QColor("red"))
                self.tableWidget.setItem(i, j, item)


    def display_table_describe(self):

        data_type = pd.DataFrame(self.dataFrame.dtypes).transpose()
        data_describe = self.dataFrame.describe()
        data_describe = pd.concat([data_type,data_describe]).replace(nan, "N/A")
        index = [str(i) for i in data_describe.index]
        index[0] = "data type"

        self.tableWidget_describe.setRowCount(data_describe.shape[0])
        self.tableWidget_describe.setColumnCount(data_describe.shape[1])
        self.tableWidget_describe.setVerticalHeaderLabels(index)
        # self.tableWidget_describe.verticalHeader().setDefaultSectionSize(30)
        # self.tableWidget_describe.verticalHeader().setResizeContentsPrecision(100)

        for i in range(data_describe.shape[0]):
            for j in range(data_describe.shape[1]):
                item = QTableWidgetItem(str(data_describe.iloc[i, j]))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if data_describe.iloc[i, j] is "N/A":
                    item.setBackground(QColor("orange"))

                self.tableWidget_describe.setItem(i, j, item)

        self.parentWidget().state_changed_handler("finish")



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