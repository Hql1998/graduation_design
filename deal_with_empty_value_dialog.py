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
        self.data = self.parentWidget().data
        # self.parent().dataFrame.
        self.setStyleSheet("background:None;")

        self.setupUi(self)
        self.update_data()

    def counting_empty(self):
        empty_num = self.data["train_x"].isnull().values.sum()
        if self.data["test_x"] is not None:
            empty_num += self.data["test_x"].isnull().values.sum()
        self.empty_cell_num_label.setText(str(empty_num))

    def update_data(self):

        if self.data["train_y"] is not None:
            data = self.data["train_x"].merge(self.data["train_y"], left_index=True, right_index=True).replace(nan,'N/A')
        else:
            data = self.data["train_x"].replace(nan, 'N/A')

        if data.shape[0] * data.shape[1] > 7000:
            data = data.head(20)
        header_data = data.columns.to_list()
        header_data[-1] = header_data[-1] + "(target)"
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
                if j == data.shape[1]-1:
                    item.setBackground(QColor("orange"))
                self.tableWidget.setItem(i, j, item)

        self.counting_empty()

    def apply_handler(self):
        print("apply clicked")

        train_x = self.data["train_x"]
        train_y = self.data["train_y"]
        test_x = self.data["test_x"]
        test_y = self.data["test_y"]

        print("good0")

        if self.drop_feature_by_threshold_cb.checkState():
            threshold_text = self.threshold_sb.text().strip()
            try:
                threshold = (100 - int(threshold_text)) * train_x.shape[0] * 0.01
                if test_x is not None:
                    data = train_x.append(test_x)
                    data = data.dropna(axis=1, thresh=threshold)
                    train_x = data.iloc[0:train_x.shape[0], :]
                    test_x = data.iloc[train_x.shape[0]:, :]
                    print(train_x.shape, test_x.shape)
                else:
                    train_x = train_x.dropna(axis=1, thresh=threshold)
                    print(train_x.shape)

            except ValueError as e:
                QErrorMessage.qtHandler()
                qErrnoWarning("invalid input at drop na by threshold line edit")


        if self.drop_empty_case_cb.checkState():
            if train_y is not None:
                data_train = train_x.merge(train_y,left_index=True, right_index=True).dropna(axis=0, how="any")
                train_y = data_train.iloc[:, -1]
                train_x = data_train.iloc[:, 0:-1]
            else:
                train_x = train_x.dropna(axis=0, how="any")

            if test_x is not None:
                if test_y is not None:
                    data_test = test_x.merge(test_y, left_index=True, right_index=True).dropna(axis=0, how="any")
                    test_y = data_test.iloc[:, -1]
                    test_x = data_test.iloc[:, 0:-1]
                else:
                    test_x = test_x.dropna(axis=0, how="any")


        text_feature_name_list = list(train_x.dtypes[train_x.dtypes == numpy.object].to_dict().keys())
        if text_feature_name_list == []:
            train_x_numeric = train_x
            if test_x is not None:
                test_x_numeric = test_x
        else:
            train_x_numeric = train_x.drop(columns=text_feature_name_list)
            if test_x is not None:
                test_x_numeric = test_x.drop(columns=text_feature_name_list)

        train_x_text = train_x.loc[:, text_feature_name_list]
        if test_x is not None:
            test_x_text = test_x.loc[:, text_feature_name_list]

        if self.fill_num_empty_with_cb.checkState() and train_x_numeric.shape[0] > 0:
            fill_way = self.fill_num_with_comb.currentText()
            simple_imputer = SimpleImputer(strategy=fill_way)
            train_x_numeric_numpy = simple_imputer.fit_transform(train_x_numeric.to_numpy())
            train_x_numeric = pd.DataFrame(train_x_numeric_numpy, index=train_x_numeric.index,
                                           columns=train_x_numeric.columns)
            if test_x is not None:
                test_x_numeric_numpy = simple_imputer.transform(test_x_numeric.to_numpy())
                test_x_numeric = pd.DataFrame(test_x_numeric_numpy, index=test_x_numeric.index,
                                              columns=test_x_numeric.columns)


        if self.fill_text_with_cb.checkState() and train_x_text.shape[1] > 0:
            fill_way = self.fill_text_with_comb.currentText()
            if fill_way == "most_frequent":
                simple_imputer = SimpleImputer(strategy="most_frequent")
            else:
                simple_imputer = SimpleImputer(strategy="constant", fill_value=fill_way)
            train_x_text_numpy = simple_imputer.fit_transform(train_x_text.to_numpy())
            train_x_text = pd.DataFrame(train_x_text_numpy, index=train_x_text.index, columns=train_x_text.columns)
            if test_x is not None:
                test_x_text_numpy = simple_imputer.transform(test_x_text.to_numpy())
                test_x_text = pd.DataFrame(test_x_text_numpy, index=test_x_text.index, columns=test_x_text.columns)


        train_x = train_x_text.merge(train_x_numeric, left_index=True, right_index=True)
        if test_x is not None:
            test_x = test_x_text.merge(test_x_numeric, left_index=True, right_index=True)

        self.data["train_x"] = train_x
        self.data["train_y"] = train_y
        self.data["test_x"] = test_x
        self.data["test_y"] = test_y

        qApp.main_window.log_te.append("\n" + "=" * 10 + self.parentWidget().class_name + "=" * 10)
        if train_x is not None:
            qApp.main_window.log_te.append("\n" + "train_x" + str(train_x.shape))
        if train_y is not None:
            qApp.main_window.log_te.append("\n" + "train_y" + str(type(train_y)))
        if test_x is not None:
            qApp.main_window.log_te.append("\n" + "test_x" + str(test_x.shape))
        if test_y is not None:
            qApp.main_window.log_te.append("\n" + "test_y" + str(type(test_y)))

        self.parentWidget().data = self.data
        self.update_data()
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
    window = Deal_With_Empty_Dialog()
    window.show()
    sys.exit(app.exec_())
