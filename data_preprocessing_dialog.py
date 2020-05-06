from GUI_part.data_preprocessing_ui import Ui_data_preprocessing
from PyQt5.Qt import *
from function_widget import *
import pandas as pd
from numpy import nan
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,MinMaxScaler,RobustScaler


class Data_Preprocessing_Dialog(QDialog, Ui_data_preprocessing):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setWindowModality(Qt.NonModal)
        self.data = self.parentWidget().data
        self.tow_nominal_var_list = []
        # self.setModal(False)
        self.setupUi(self)
        self.update_data()
        self.parentWidget().state_changed_handler("start")


    def update_data(self):

        if self.data["train_y"] is not None:
            data = self.data["train_x"].merge(self.data["train_y"], left_index=True, right_index=True).replace(nan, 'N/A')
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
                if j == data.shape[1] - 1:
                    item.setBackground(QColor("orange"))
                self.tableWidget.setItem(i, j, item)


    def parameter_changed_handler(self):
       if self.save_file_cb.checkState():
           self.save_file_btn.setEnabled(True)
           self.save_file_btn.click()
       else:
           self.save_file_btn.setEnabled(False)

    def save_file_btn_clicked_handler(self):

        if self.save_file_btn.open_result[0] != "":
            self.save_file_label.setText("/".join(self.save_file_btn.open_result[0].split("/")[0:-1]))
            self.save_file_label.adjustSize()
        else:
            self.save_file_label.setText("No Directory selected")
            self.save_file_label.adjustSize()


    def apply_handler(self):
        print("apply clicked")
        self.parentWidget().state_changed_handler("processing")

        train_x = self.data["train_x"]
        test_x = self.data["test_x"]

        # 处理等级编码
        ordinal_index_text = self.ordinal_index_le.text()
        ordinal_index_list = []
        train_ordinal_data = None
        test_ordinal_data = None
        if self.trans_into_ordinal_cb.checkState() and len(ordinal_index_text) > 0:
            try:
                for i in ordinal_index_text.split(","):
                    i = i.strip()
                    ordinal_index_list.append(int(i))
                train_ordinal_data = train_x.iloc[:, ordinal_index_list]
                ordinal_encoder = OrdinalEncoder()
                train_ordinal_data_numpy = ordinal_encoder.fit_transform(train_ordinal_data.to_numpy())
                train_ordinal_data = pd.DataFrame(train_ordinal_data_numpy, index=train_ordinal_data.index, columns=train_ordinal_data.columns)
                if test_x is not None:
                    test_ordinal_data = test_x.iloc[:, ordinal_index_list]
                    test_ordinal_data_numpy = ordinal_encoder.transform(test_ordinal_data.to_numpy())
                    test_ordinal_data = pd.DataFrame(test_ordinal_data_numpy, index=test_ordinal_data.index,
                                                      columns=test_ordinal_data.columns)
            except:
                print("invalid input")
                QErrorMessage.qtHandler()
                qErrnoWarning("invalid input at ordinal transforming index line edit")

        # 处理onehot编码
        onehot_index_text = self.onehot_index_le.text()
        onehot_index_list = []
        train_onehot_data = None
        test_onehot_data = None
        if self.trans_into_onehot_cb.checkState() and len(onehot_index_text) > 0:
            try:
                for i in onehot_index_text.split(","):
                    i = i.strip()
                    onehot_index_list.append(int(i))
                train_onehot_data = train_x.iloc[:, onehot_index_list]
                onehot_encoder = OneHotEncoder(sparse=False)
                train_onehot_data_numpy = onehot_encoder.fit_transform(train_onehot_data.to_numpy())
                onehot_columns = []
                for i, v in enumerate(train_onehot_data.columns):
                    for j in onehot_encoder.categories_[i]:
                        onehot_columns.append(v + "_" + str(j))
                train_onehot_data = pd.DataFrame(train_onehot_data_numpy, index=train_onehot_data.index, columns=onehot_columns)

                if test_x is not None:
                    test_onehot_data = test_x.iloc[:, onehot_index_list]
                    test_onehot_data_numpy = onehot_encoder.transform(test_onehot_data.to_numpy())
                    test_onehot_data = pd.DataFrame(test_onehot_data_numpy, index=test_onehot_data.index,
                                                     columns=onehot_columns)

            except KeyError as e:
                print("invalid input")
                QErrorMessage.qtHandler()
                qErrnoWarning("invalid input at onehot 编码" + str(e))

        if len(ordinal_index_list + onehot_index_list) > 0:
            index_list = ordinal_index_list + onehot_index_list
            labels = train_x.columns[index_list]
            train_x = train_x.drop(columns=labels)
            if test_x is not None:
                test_x = test_x.drop(columns=labels)


        # 处理drop_index
        drop_index_text = self.drop_index_le.text()
        if self.drop_feature_by_index_cb.checkState() and len(drop_index_text) > 0:
            try:
                index_list = []
                for i in drop_index_text.split(","):
                    i = i.strip()
                    index_list.append(int(i))
                labels = train_x.columns[index_list]
                train_x = train_x.drop(columns=labels)
                if test_x is not None:
                    test_x = test_x.drop(columns=labels)
            except:
                QErrorMessage.qtHandler()
                qErrnoWarning("invalid input at drop feature by index line edit")

        if train_ordinal_data is not None:
            train_x = train_x.merge(train_ordinal_data, left_index=True, right_index=True)
        if train_onehot_data is not None:
            train_x = train_x.merge(train_onehot_data, left_index=True, right_index=True)
        if test_x is not None:
            if test_ordinal_data is not None:
                test_x = test_x.merge(test_ordinal_data, left_index=True, right_index=True)
            if test_onehot_data is not None:
                test_x = test_x.merge(test_onehot_data, left_index=True, right_index=True)

        if self.scaled_cb.checkState():
            scale_way = self.scale_by_comb.currentText()
            self.tow_nominal_var_list = []
            for col in train_x.columns:
                if len(train_x[col].unique().tolist()) == 2:
                    print(col, train_x[col].unique().tolist())
                    self.tow_nominal_var_list.append(col)

            if self.tow_nominal_var_list != []:
                train_x_need_change = train_x.drop(columns = self.tow_nominal_var_list)
                if test_x is not None:
                    test_x_need_change = test_x.drop(columns=self.tow_nominal_var_list)
            else:
                train_x_need_change = train_x
                if test_x is not None:
                    test_x_need_change = test_x

            if scale_way == "RobustScaler":
                RS = RobustScaler()
                train_x_scaled = RS.fit_transform(train_x_need_change.to_numpy())
                if test_x is not None:
                    test_x_scaled = RS.transform(test_x_need_change.to_numpy())
            elif scale_way == "MinMaxScaler":
                MMS = MinMaxScaler()
                train_x_scaled = MMS.fit_transform(train_x_need_change.to_numpy())
                if test_x is not None:
                    test_x_scaled = MMS.transform(test_x_need_change.to_numpy())
            elif scale_way == "StandardScaler":
                SS = StandardScaler()
                train_x_scaled = SS.fit_transform(train_x_need_change.to_numpy())
                if test_x is not None:
                    test_x_scaled = SS.transform(test_x_need_change.to_numpy())

            train_x_need_change = pd.DataFrame(train_x_scaled, index=train_x_need_change.index, columns=train_x_need_change.columns)
            if test_x is not None:
                test_x_need_change = pd.DataFrame(test_x_scaled, index=test_x_need_change.index, columns=test_x_need_change.columns)

            if self.tow_nominal_var_list != []:
                train_x = train_x_need_change.merge(train_x.loc[:, self.tow_nominal_var_list],left_index=True,right_index=True)
                if test_x is not None:
                    test_x = train_x_need_change.merge(test_x.loc[:, self.tow_nominal_var_list],left_index=True,right_index=True)
            else:
                train_x = train_x_need_change
                if test_x is not None:
                    test_x = train_x_need_change


        self.data["train_x"] = train_x
        self.data["test_x"] = test_x

        qApp.main_window.log_te.append("\n" + "=" * 20 + self.parentWidget().class_name + "=" * 20)
        if self.data["train_x"] is not None:
            qApp.main_window.log_te.append("\n" + "train_x" + str(self.data["train_x"].shape))
        if self.data["train_y"] is not None:
            qApp.main_window.log_te.append("\n" + "train_y" + str(type(self.data["train_y"])))
        if self.data["test_x"] is not None:
            qApp.main_window.log_te.append("\n" + "test_x" + str(self.data["test_x"].shape))
        if self.data["test_y"] is not None:
            qApp.main_window.log_te.append("\n" + "test_y" + str(type(self.data["test_y"])))

        qApp.main_window.log_te.append("\n" + "0-1 columns :" + ", ".join(self.tow_nominal_var_list))

        if self.save_file_cb.checkState() and self.save_file_label.text() != "No Directory selected":
            if self.data["train_y"] is not None:
                train_data = self.data["train_x"].merge(self.data["train_y"], left_index=True, right_index=True)
                if self.data["test_x"] is not None:
                    test_data = self.data["test_x"].merge(self.data["test_y"], left_index=True, right_index=True)
            else:
                train_data = self.data["train_x"]
                if self.data["test_x"] is not None:
                    test_data = self.data["test_x"]

            file_path = self.save_file_btn.open_result[0]

            if self.save_file_btn.open_result[1] == "csv(*.csv)":
                print("csv ",file_path)
                train_data.to_csv(file_path.replace(".csv","_train_{0}.csv".format(self.parentWidget().class_name)))
                if self.data["test_x"] is not None:
                    test_data.to_csv(file_path.replace(".csv", "_test_{0}.csv".format(self.parentWidget().class_name)))
            elif self.save_file_btn.open_result[1] == "excel(*.xlsx)":
                train_data.to_excel(file_path.replace(".xlsx", "_train_{0}.xlsx".format(self.parentWidget().class_name)))
                if self.data["test_x"] is not None:
                    test_data.to_excel(file_path.replace(".xlsx", "_test_{0}.xlsx".format(self.parentWidget().class_name)))
            else:
                QErrorMessage.qtHandler()
                qErrnoWarning("invalid save file name")

        self.parentWidget().data = self.data
        self.update_data()

        if self.parent().next_widgets != [] and self.parent().next_widgets[0].data is None:
            self.parent().next_widgets[0].update_data_from_previous()

        self.parentWidget().state_changed_handler("finish")

    def finish_handler(self):
        print("finished")
        if self.parent().next_widgets != [] and self.parent().next_widgets[0].data is None:
            self.parent().next_widgets[0].update_data_from_previous()
        self.hide()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Data_Preprocessing_Dialog()
    window.show()
    sys.exit(app.exec_())
