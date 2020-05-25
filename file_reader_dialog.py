from GUI_part.file_reader_dialog_ui import Ui_file_reader_dialog
from print_to_log import *
from function_widget import *
import pandas as pd
from numpy import nan
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


class File_Reader_Dialog(QDialog,Ui_file_reader_dialog):

    state_changed = pyqtSignal(str)

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setupUi(self)
        self.setup_ui_subtle()
        # self.open_file_btn.click()
        self.setObjectName("file_reader_dialog")
        self.dataFrame_train = None
        self.dataFrame_test = None
        self.dataFrame_test_loaded = None
        self.destroyed.connect(lambda: print("File_Reader_Dialog were destroyed"))

    def setup_ui_subtle(self):

        btn_group = QButtonGroup(self)
        btn_group.addButton(self.load_test_from_cb, 1)
        btn_group.addButton(self.split_test_fro_train_cb, 2)
        btn_group.addButton(self.without_test_cb, 3)

        self.load_test_from_cb.toggled.connect(self.load_test_handler)
        self.train_column_header_cb.toggled.connect(self.updata_train_header)
        self.train_row_header_cb.toggled.connect(self.updata_train_header)
        self.test_column_header_cb.toggled.connect(self.updata_test_header)
        self.test_row_header_cb.toggled.connect(self.updata_test_header)

    def load_test_handler(self):
        checked = self.load_test_from_cb.checkState()
        if checked:
            self.open_test_file_btn.setEnabled(True)
            self.open_test_file_btn.click()
        else:
            self.open_test_file_btn.setEnabled(False)
            self.dataFrame_test_loaded = None
            self.display_test_file_name_label.setText("No File Selected")
            self.display_test_file_name_label.adjustSize()

    def update_target_comb(self, target_index=-1):

        header_data = self.dataFrame_train.columns.to_list()
        self.target_index_comb.clear()
        self.target_index_comb.addItems(header_data)
        if target_index == -1:
            self.target_index_comb.setCurrentIndex(len(header_data) - 1)
        else:
            self.target_index_comb.setCurrentIndex(target_index)

    def open_test_file_btn_clicked_handler(self):

        if self.open_test_file_btn.open_result[0] != "":

            self.display_test_file_name_label.setText(self.open_test_file_btn.open_result[0].split("/")[-1])
            self.display_test_file_name_label.adjustSize()
            self.read_test_data()

    def read_test_data(self):

        self.parentWidget().state_changed_handler("processing")
        if self.test_row_header_cb.checkState():
            index_col = 0
        else:
            index_col = None
        if self.test_column_header_cb.checkState():
            header = 0
        else:
            header = None
        if self.open_test_file_btn.open_result[1] == "excel(*.xlsx)":
            self.dataFrame_test_loaded = pd.read_excel(self.open_test_file_btn.open_result[0], header=header, index_col=index_col)
        elif self.open_test_file_btn.open_result[1] == "csv(*.csv)":
            self.dataFrame_test_loaded = pd.read_csv(self.open_test_file_btn.open_result[0], header=header, index_col=index_col)
        elif self.open_test_file_btn.open_result[1] == "tsv(*.tsv)":
            self.dataFrame_test_loaded = pd.read_csv(self.open_test_file_btn.open_result[0], sep="\t", header=header, index_col=index_col)

    def updata_test_header(self):
        if self.dataFrame_test_loaded is not None:
            self.read_test_data()

    def updata_train_header(self):
        if self.dataFrame_train is not None:
            self.open_train_file_btn_clicked_handler()

    def open_train_file_btn_clicked_handler(self):
        if self.open_train_file_btn.open_result[0] != "":
            self.parentWidget().state_changed_handler("start")
            self.display_train_file_name_label.setText(self.open_train_file_btn.open_result[0].split("/")[-1])
            self.display_train_file_name_label.adjustSize()

            self.load_test_from_cb.setEnabled(True)
            self.split_test_fro_train_cb.setEnabled(True)
            self.without_test_cb.setEnabled(True)
            self.set_target_cb.setEnabled(True)
            self.apply_btn.setEnabled(True)

            self.read_train_data()
            self.update_target_comb(target_index=self.dataFrame_train.shape[1] - 1)
            self.display_table()
            self.display_table_describe()

    def target_label_changed_handler(self,index):
        print("target changed")
        if self.set_target_cb.checkState():
            self.display_table()


    def read_train_data(self):

        self.parentWidget().state_changed_handler("processing")
        if self.train_row_header_cb.checkState():
            index_col = 0
        else:
            index_col = None
        if self.train_column_header_cb.checkState():
            header = 0
        else:
            header = None
        if self.open_train_file_btn.open_result[1] == "excel(*.xlsx)":
            self.dataFrame_train = pd.read_excel(self.open_train_file_btn.open_result[0], header=header, index_col=index_col)
        elif self.open_train_file_btn.open_result[1] == "csv(*.csv)":
            self.dataFrame_train = pd.read_csv(self.open_train_file_btn.open_result[0], header=header, index_col=index_col)
        elif self.open_train_file_btn.open_result[1] == "tsv(*.tsv)":
            self.dataFrame_train = pd.read_csv(self.open_train_file_btn.open_result[0], sep="\t", header=header, index_col=index_col)

    def display_table(self):

        data = self.dataFrame_train.replace(nan, 'N/A')
        if data.shape[0] * data.shape[1] > 7000:
            data = data.head(20)
        header_data = data.columns.to_list()

        if self.set_target_cb.checkState():
            target_index = self.target_index_comb.currentIndex()
            header_data[target_index] = header_data[target_index] + " (target)"

        self.tableWidget.setRowCount(data.shape[0])
        self.tableWidget.setColumnCount(data.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(header_data)
        self.tableWidget.setVerticalHeaderLabels([str(i) for i in data.index])

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = QTableWidgetItem(str(data.iloc[i, j]))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if data.iloc[i, j] == 'N/A':
                    item.setBackground(QColor("red"))
                if self.set_target_cb.checkState() and j == target_index:
                    item.setBackground(QColor("orange"))
                self.tableWidget.setItem(i, j, item)

    def display_table_describe(self):

        data_type = pd.DataFrame(self.dataFrame_train.dtypes).transpose()
        data_describe = self.dataFrame_train.describe()
        data_describe = pd.concat([data_type, data_describe]).replace(nan, "N/A")
        index = [str(i) for i in data_describe.index]
        index[0] = "data type"

        self.tableWidget_describe.setRowCount(data_describe.shape[0])
        self.tableWidget_describe.setColumnCount(data_describe.shape[1])
        self.tableWidget_describe.setVerticalHeaderLabels(index)

        for i in range(data_describe.shape[0]):
            for j in range(data_describe.shape[1]):
                item = QTableWidgetItem(str(data_describe.iloc[i, j]))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if data_describe.iloc[i, j] is "N/A":
                    item.setBackground(QColor("orange"))

                self.tableWidget_describe.setItem(i, j, item)

    def apply_handler(self):

        if self.split_test_fro_train_cb.checkState():
            self.dataFrame_test = None
            self.read_train_data()

            test_ratio = self.test_radio.value() / 100
            split_method = self.split_method_comb.currentText()
            if split_method == "Random Split":
                train_set, test_set = train_test_split(self.dataFrame_train, test_size=test_ratio , random_state=42)
                self.dataFrame_train = train_set
                self.dataFrame_test = test_set
            elif split_method == "Stratified Shuffle Split":
                SSsplit = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
                target_index = self.target_index_comb.currentIndex()
                for train_index, test_index in SSsplit.split(self.dataFrame_train, self.dataFrame_train.iloc[:, target_index]):
                    strat_train_set = self.dataFrame_train.loc[train_index]
                    strat_test_set = self.dataFrame_train.loc[test_index]
                self.dataFrame_train = strat_train_set
                self.dataFrame_test = strat_test_set
            self.display_table()
            self.display_table_describe()
        if self.load_test_from_cb.checkState():
            if self.dataFrame_test_loaded is None:
                QErrorMessage.qtHandler()
                qErrnoWarning("please select a test file")
                return None
            else:
                if self.dataFrame_test_loaded.shape[1] != self.dataFrame_train.shape[1]:
                    QErrorMessage.qtHandler()
                    qErrnoWarning("please select a validate test file")
                    return None
                else:
                    self.dataFrame_test = self.dataFrame_test_loaded
        if self.without_test_cb.checkState():
            self.dataFrame_test = None


        self.parentWidget().data={}
        print_log_header(self.parentWidget().class_name)

        if self.set_target_cb.checkState():

            target_index = self.target_index_comb.currentIndex()
            self.parentWidget().data["train_y"] = self.dataFrame_train.iloc[:, target_index]
            self.parentWidget().data["train_x"] = self.dataFrame_train.drop(columns=self.dataFrame_train.columns[target_index])
            qApp.main_window.log_te.append("\n" + "train_x's shape " +str(self.parentWidget().data["train_x"].shape)+  str(type(self.parentWidget().data["train_x"])))
            qApp.main_window.log_te.append("\n" + "train_y's shape " + str(type(self.parentWidget().data["train_y"])))
            if self.dataFrame_test is not None:
                self.parentWidget().data["test_y"] = self.dataFrame_test.iloc[:, target_index]
                self.parentWidget().data["test_x"] = self.dataFrame_test.drop(
                    columns=self.dataFrame_test.columns[target_index])
                qApp.main_window.log_te.append(
                    "\n" + "test_x's shape " + str(self.parentWidget().data["test_x"].shape))
                qApp.main_window.log_te.append(
                    "\n" + "test_y's shape " + str(type(self.parentWidget().data["test_y"])))
            else:
                self.parentWidget().data["test_y"] = None
                self.parentWidget().data["test_x"] = None
                qApp.main_window.log_te.append("\n" + "test_x's shape " + "None")
                qApp.main_window.log_te.append("\n" + "test_y's shape " + "None")
        else:

            self.parentWidget().data["train_y"] = None
            self.parentWidget().data["train_x"] = self.dataFrame_train
            qApp.main_window.log_te.append("\n" + "train_x's shape " + str(self.parentWidget().data["train_x"].shape))
            qApp.main_window.log_te.append("\n" + "train_y's shape " + "None")

            if self.dataFrame_test is not None:
                self.parentWidget().data["test_y"] = None
                self.parentWidget().data["test_x"] = self.dataFrame_test
                qApp.main_window.log_te.append("\n" + "test_x's shape " + str(self.parentWidget().data["test_x"].shape))
                qApp.main_window.log_te.append("\n" + "test_y's shape " + "None")
            else:
                self.parentWidget().data["test_y"] = None
                self.parentWidget().data["test_x"] = None
                qApp.main_window.log_te.append("\n" + "test_x's shape " + "None")
                qApp.main_window.log_te.append("\n" + "test_y's shape " + "None")

        if self.parent().next_widgets != [] and self.parent().next_widgets[0].data is None:
            self.parent().next_widgets[0].update_data_from_previous()

        self.parentWidget().state_changed_handler("finish")

    def finish_handler(self):

        if self.parent().next_widgets != []:
            for next_widget in self.parent().next_widgets:
                if next_widget.data is None:
                    next_widget.update_data_from_previous()
        self.hide()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = File_Reader_Dialog()
    window.show()
    sys.exit(app.exec_())