from GUI_part.deal_with_empty_value_ui import Ui_Deal_With_Empty_Dialog

from function_widget import *
import pandas as pd
from numpy import nan
import numpy
from sklearn.impute import SimpleImputer


class Deal_With_Empty_Dialog(QDialog, Ui_Deal_With_Empty_Dialog):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setWindowModality(Qt.NonModal)
        # self.dataFrame = pd.read_excel("./temp/radiomic_test_data.xlsx")
        # self.parent().dataFrame.
        self.setStyleSheet("background:None;")

        self.setupUi(self)
        self.update_data()

    def counting_empty(self):
        empty_num = self.parent().dataFrame.isnull().values.sum()
        # empty_num = self.dataFrame.isnull().values.sum()
        self.empty_cell_num_label.setText(str(empty_num))

    def update_data(self):

        data = self.parent().dataFrame.replace(nan, 'N/A')
        # data = self.dataFrame.replace(nan, 'N/A')
        header_data = data.columns.to_list()
        for index, value in enumerate(header_data):
            header_data[index] = str(index) + "\n" + header_data[index]

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

        self.counting_empty()

    def apply_handler(self):
        print("apply clicked")

        data = self.parent().dataFrame

        if self.drop_feature_by_threshold_cb.checkState():
            threshold_text = self.threshold_sb.text().strip()
            try:
                threshold = (100 - int(threshold_text)) * data.shape[0] * 0.01
                data = data.dropna(axis=1, thresh=threshold)
            except ValueError as e:
                QErrorMessage.qtHandler()
                qErrnoWarning("invalid input at drop na by threshold line edit")

        if self.drop_empty_case_cb.checkState():
            data = data.dropna(axis=0, how="any")

        text_feature_name_list = list(data.dtypes[data.dtypes == numpy.object].to_dict().keys())
        if text_feature_name_list == []:
            data_num = data
        else:
            data_num = data.drop(columns=text_feature_name_list)
        data_text = data.loc[:, text_feature_name_list]

        if self.fill_num_empty_with_cb.checkState() and data_text.shape[0] > 0:
            fill_way = self.fill_num_with_comb.currentText()
            simple_imputer = SimpleImputer(strategy=fill_way)
            data_numpy = simple_imputer.fit_transform(data_num.to_numpy())
            data_num = pd.DataFrame(data_numpy, index=data_num.index, columns=data_num.columns)

        if self.fill_text_with_cb.checkState() and data_text.shape[1] > 0:
            fill_way = self.fill_text_with_comb.currentText()
            if fill_way == "most_frequent":
                simple_imputer = SimpleImputer(strategy="most_frequent")
            else:
                simple_imputer = SimpleImputer(strategy="constant", fill_value=fill_way)
            data_numpy = simple_imputer.fit_transform(data_text.to_numpy())
            data_text = pd.DataFrame(data_numpy, index=data_text.index, columns=data_text.columns)

        data = data_text.merge(data_num, left_index=True, right_index=True)
        self.parent().dataFrame = data
        self.update_data()
        self.parentWidget().state_changed_handler("finish")

    def finish_handler(self):
        if self.parent().next_widgets != [] and self.parent().next_widgets[0].dataFrame is None:
            self.parent().next_widgets[0].update_data_from_previous()
        self.hide()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Deal_With_Empty_Dialog()
    window.show()
    sys.exit(app.exec_())
