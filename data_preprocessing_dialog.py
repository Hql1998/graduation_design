from GUI_part.data_preprocessing_ui import Ui_data_preprocessing
from PyQt5.Qt import *
from function_widget import *
import pandas as pd
from numpy import nan
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,MinMaxScaler,RobustScaler
from sklearn.impute import SimpleImputer


class Data_Preprocessing_Dialog(QDialog, Ui_data_preprocessing):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setWindowModality(Qt.NonModal)
        # self.dataFrame = pd.read_csv("./temp/balanced_diabetes_dataset.csv")
        # self.setModal(False)
        self.setupUi(self)
        self.update_data()

    def update_target_comb(self, target_index = -1):
        # header_data = self.dataFrame.columns.to_list()
        header_data = self.parent().dataFrame.columns.to_list()
        self.target_index_comb.blockSignals(True)
        self.target_index_comb.clear()
        self.target_index_comb.addItems(header_data)
        if target_index == -1:
            self.target_index_comb.setCurrentIndex(len(header_data) - 1)
        else:
            self.target_index_comb.setCurrentIndex(target_index)
        self.target_index_comb.blockSignals(False)

    def update_data(self):

        self.update_target_comb(target_index=self.parentWidget().dataFrame.shape[1] - 1)
        data = self.parent().dataFrame.replace(nan, 'N/A')
        # data = self.dataFrame.replace(nan, 'N/A')
        header_data = data.columns.to_list()
        for index, value in enumerate(header_data):
            header_data[index] = str(index) + "\n" + header_data[index]

        if self.set_target_cb.checkState():
            target_index = self.target_index_comb.currentIndex()
            header_data[target_index] = header_data[target_index] + " (target)"

        self.tableWidget.setRowCount(data.shape[0])
        self.tableWidget.setColumnCount(data.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(header_data)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = QTableWidgetItem(str(data.iloc[i, j]))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if data.iloc[i, j] == 'N/A':
                    item.setBackground(QColor("red"))
                if self.set_target_cb.checkState() and j == target_index:
                    item.setBackground(QColor("orange"))
                self.tableWidget.setItem(i, j, item)


    def parameter_changed_handler(self):
        print("参数改变了")



    def apply_handler(self):
        print("apply clicked")
        data = self.parentWidget().dataFrame

        target_index = self.target_index_comb.currentIndex()
        if self.set_target_cb.checkState():
            data_y = data.iloc[:, target_index]
            data_x = data.drop(columns=data.columns[target_index])
            data = data_x

        # 处理等级编码
        ordinal_index_text = self.ordinal_index_le.text()
        ordinal_index_list = []
        ordinal_data = None
        if self.trans_into_ordinal_cb.checkState() and len(ordinal_index_text) > 0:
            try:
                for i in ordinal_index_text.split(","):
                    i = i.strip()
                    ordinal_index_list.append(int(i))
                ordinal_data = data.iloc[:, ordinal_index_list]
                ordinal_encoder = OrdinalEncoder()
                ordinal_data_numpy = ordinal_encoder.fit_transform(ordinal_data.to_numpy())
                ordinal_data = pd.DataFrame(ordinal_data_numpy, index=ordinal_data.index, columns=ordinal_data.columns)

            except:
                print("invalid input")
                QErrorMessage.qtHandler()
                qErrnoWarning("invalid input at ordinal transforming index line edit")

        # 处理onehot编码
        onehot_index_text = self.onehot_index_le.text()
        onehot_index_list = []
        onehot_data = None
        if self.trans_into_onehot_cb.checkState() and len(onehot_index_text) > 0:
            try:
                for i in onehot_index_text.split(","):
                    i = i.strip()
                    onehot_index_list.append(int(i))
                onehot_data = data.iloc[:, onehot_index_list]
                onehot_encoder = OneHotEncoder(sparse=False)
                onehot_data_numpy = onehot_encoder.fit_transform(onehot_data.to_numpy())
                onehot_columns = []
                for i, v in enumerate(onehot_data.columns):
                    for j in onehot_encoder.categories_[i]:
                        onehot_columns.append(v + "_" + str(j))
                onehot_data = pd.DataFrame(onehot_data_numpy, index=onehot_data.index, columns=onehot_columns)
            except KeyError as e:
                print("invalid input")
                QErrorMessage.qtHandler()
                qErrnoWarning("invalid input at onehot 编码" + str(e))

        # 处理drop_index
        drop_index_text = self.drop_index_le.text()
        if self.drop_feature_by_index_cb.checkState() and len(drop_index_text) > 0:
            try:
                index_list = []
                for i in drop_index_text.split(","):
                    i = i.strip()
                    index_list.append(int(i))
                index_list = index_list + ordinal_index_list + onehot_index_list
                labels = data.columns[index_list]
                data = data.drop(columns=labels)
            except:
                QErrorMessage.qtHandler()
                qErrnoWarning("invalid input at drop feature by index line edit")
                return None
        if ordinal_data is not None:
            data = data.merge(ordinal_data, left_index=True, right_index=True)
        if onehot_data is not None:
            data = data.merge(onehot_data, left_index=True, right_index=True)

        if self.scaled_cb.checkState():
            scale_way = self.scale_by_comb.currentText()
            if scale_way == "RobustScaler":
                data_scaled = RobustScaler().fit_transform(data.to_numpy())
            elif scale_way == "MinMaxScaler":
                data_scaled = MinMaxScaler().fit_transform(data.to_numpy())
            elif scale_way == "StandardScaler":
                data_scaled = StandardScaler().fit_transform(data.to_numpy())
            data = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
            print(scale_way)

        data = data.merge(data_y, left_index=True, right_index=True)
        self.parentWidget().dataFrame = data
        self.update_data()

        if self.parent().next_widgets != [] and self.parent().next_widgets[0].dataFrame is None:
            self.parent().next_widgets[0].update_data_from_previous()

        self.parentWidget().state_changed_handler("finish")

    def finish_handler(self):
        print("finished")
        if self.parent().next_widgets != [] and self.parent().next_widgets[0].dataFrame is None:
            self.parent().next_widgets[0].update_data_from_previous()
        self.hide()



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Data_Preprocessing_Dialog()
    window.show()
    sys.exit(app.exec_())
