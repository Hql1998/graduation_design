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
        # self.setModal(False)
        self.setupUi(self)
        self.update_data()


    def update_data(self):

        if self.data["train_y"] is not None:
            data = self.data["train_x"].merge(self.data["train_y"], left_index=True, right_index=True).replace(nan, 'N/A')
        else:
            data = self.data["train_x"].replace(nan, 'N/A')

        if data.shape[0] * data.shape[1] > 10000:
            data = data.head(30)
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
        print("参数改变了")



    def apply_handler(self):
        print("apply clicked")
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

            if scale_way == "RobustScaler":
                RS = RobustScaler()
                train_x_scaled = RS.fit_transform(train_x.to_numpy())
                if test_x is not None:
                    test_x_scaled = RS.transform(test_x.to_numpy())
            elif scale_way == "MinMaxScaler":
                MMS = MinMaxScaler()
                train_x_scaled = MMS.fit_transform(train_x.to_numpy())
                if test_x is not None:
                    test_x_scaled = MMS.transform(test_x.to_numpy())
            elif scale_way == "StandardScaler":
                SS = StandardScaler()
                train_x_scaled = SS.fit_transform(train_x.to_numpy())
                if test_x is not None:
                    test_x_scaled = SS.transform(test_x.to_numpy())
            train_x = pd.DataFrame(train_x_scaled, index=train_x.index, columns=train_x.columns)
            if test_x is not None:
                test_x = pd.DataFrame(test_x_scaled, index=test_x.index, columns=test_x.columns)

        self.data["train_x"] = train_x
        self.data["test_x"] = test_x

        qApp.main_window.log_te.append("\n" + "=" * 10 + self.parentWidget().class_name + "=" * 10)
        if self.data["train_x"] is not None:
            qApp.main_window.log_te.append("\n" + "train_x" + str(self.data["train_x"].shape))
        if self.data["train_y"] is not None:
            qApp.main_window.log_te.append("\n" + "train_y" + str(type(self.data["train_y"])))
        if self.data["test_x"] is not None:
            qApp.main_window.log_te.append("\n" + "test_x" + str(self.data["test_x"].shape))
        if self.data["test_y"] is not None:
            qApp.main_window.log_te.append("\n" + "test_y" + str(type(self.data["test_y"])))

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
