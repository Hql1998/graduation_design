# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'data_preprocessing.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_data_preprocessing(object):
    def setupUi(self, data_preprocessing):
        data_preprocessing.setObjectName("data_preprocessing")
        data_preprocessing.resize(518, 739)
        data_preprocessing.setStyleSheet("background: None;")
        self.verticalLayout = QtWidgets.QVBoxLayout(data_preprocessing)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.target_label_groupBox = QtWidgets.QGroupBox(data_preprocessing)
        self.target_label_groupBox.setMinimumSize(QtCore.QSize(0, 50))
        self.target_label_groupBox.setObjectName("target_label_groupBox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.target_label_groupBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem1, 0, 3, 1, 1)
        self.target_index_comb = QtWidgets.QComboBox(self.target_label_groupBox)
        self.target_index_comb.setObjectName("target_index_comb")
        self.gridLayout_4.addWidget(self.target_index_comb, 0, 2, 1, 1)
        self.set_target_cb = QtWidgets.QCheckBox(self.target_label_groupBox)
        self.set_target_cb.setChecked(True)
        self.set_target_cb.setObjectName("set_target_cb")
        self.gridLayout_4.addWidget(self.set_target_cb, 0, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem2, 0, 1, 1, 1)
        self.verticalLayout.addWidget(self.target_label_groupBox)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem3)
        self.text_feature_groupBox = QtWidgets.QGroupBox(data_preprocessing)
        self.text_feature_groupBox.setMinimumSize(QtCore.QSize(0, 100))
        self.text_feature_groupBox.setObjectName("text_feature_groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.text_feature_groupBox)
        self.gridLayout_3.setVerticalSpacing(9)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.drop_feature_by_index_cb = QtWidgets.QCheckBox(self.text_feature_groupBox)
        self.drop_feature_by_index_cb.setChecked(True)
        self.drop_feature_by_index_cb.setObjectName("drop_feature_by_index_cb")
        self.gridLayout_3.addWidget(self.drop_feature_by_index_cb, 1, 0, 1, 1)
        self.drop_index_le = QtWidgets.QLineEdit(self.text_feature_groupBox)
        self.drop_index_le.setObjectName("drop_index_le")
        self.gridLayout_3.addWidget(self.drop_index_le, 1, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.text_feature_groupBox)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 0, 2, 1, 1)
        self.onehot_index_le = QtWidgets.QLineEdit(self.text_feature_groupBox)
        self.onehot_index_le.setObjectName("onehot_index_le")
        self.gridLayout_3.addWidget(self.onehot_index_le, 3, 2, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem4, 2, 3, 1, 1)
        self.trans_into_ordinal_cb = QtWidgets.QCheckBox(self.text_feature_groupBox)
        self.trans_into_ordinal_cb.setObjectName("trans_into_ordinal_cb")
        self.gridLayout_3.addWidget(self.trans_into_ordinal_cb, 2, 0, 1, 1)
        self.ordinal_index_le = QtWidgets.QLineEdit(self.text_feature_groupBox)
        self.ordinal_index_le.setObjectName("ordinal_index_le")
        self.gridLayout_3.addWidget(self.ordinal_index_le, 2, 2, 1, 1)
        self.trans_into_onehot_cb = QtWidgets.QCheckBox(self.text_feature_groupBox)
        self.trans_into_onehot_cb.setObjectName("trans_into_onehot_cb")
        self.gridLayout_3.addWidget(self.trans_into_onehot_cb, 3, 0, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem5, 2, 1, 1, 1)
        self.verticalLayout.addWidget(self.text_feature_groupBox)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem6)
        self.scale_groupBox = QtWidgets.QGroupBox(data_preprocessing)
        self.scale_groupBox.setMinimumSize(QtCore.QSize(0, 50))
        self.scale_groupBox.setObjectName("scale_groupBox")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.scale_groupBox)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.scaled_cb = QtWidgets.QCheckBox(self.scale_groupBox)
        self.scaled_cb.setChecked(True)
        self.scaled_cb.setObjectName("scaled_cb")
        self.gridLayout_5.addWidget(self.scaled_cb, 0, 0, 1, 1)
        self.scale_by_comb = QtWidgets.QComboBox(self.scale_groupBox)
        self.scale_by_comb.setObjectName("scale_by_comb")
        self.scale_by_comb.addItem("")
        self.scale_by_comb.addItem("")
        self.scale_by_comb.addItem("")
        self.gridLayout_5.addWidget(self.scale_by_comb, 0, 2, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem7, 0, 1, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem8, 0, 3, 1, 1)
        self.verticalLayout.addWidget(self.scale_groupBox)
        spacerItem9 = QtWidgets.QSpacerItem(20, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem9)
        self.table_groupBox = QtWidgets.QGroupBox(data_preprocessing)
        self.table_groupBox.setMinimumSize(QtCore.QSize(500, 250))
        self.table_groupBox.setObjectName("table_groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.table_groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.tableWidget = QtWidgets.QTableWidget(self.table_groupBox)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.verticalLayout_3.addWidget(self.tableWidget)
        self.verticalLayout.addWidget(self.table_groupBox)
        self.widget = QtWidgets.QWidget(data_preprocessing)
        self.widget.setMinimumSize(QtCore.QSize(0, 50))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setHorizontalSpacing(20)
        self.gridLayout.setObjectName("gridLayout")
        self.apply_btn = QtWidgets.QPushButton(self.widget)
        self.apply_btn.setObjectName("apply_btn")
        self.gridLayout.addWidget(self.apply_btn, 0, 0, 1, 1)
        self.finish_btn = QtWidgets.QPushButton(self.widget)
        self.finish_btn.setObjectName("finish_btn")
        self.gridLayout.addWidget(self.finish_btn, 0, 3, 1, 1)
        self.reset_btn = QtWidgets.QPushButton(self.widget)
        self.reset_btn.setObjectName("reset_btn")
        self.gridLayout.addWidget(self.reset_btn, 0, 2, 1, 1)
        self.verticalLayout.addWidget(self.widget)

        self.retranslateUi(data_preprocessing)
        self.drop_feature_by_index_cb.clicked['bool'].connect(data_preprocessing.parameter_changed_handler)
        self.trans_into_ordinal_cb.clicked['bool'].connect(data_preprocessing.parameter_changed_handler)
        self.trans_into_onehot_cb.clicked['bool'].connect(data_preprocessing.parameter_changed_handler)
        self.drop_index_le.textEdited['QString'].connect(data_preprocessing.parameter_changed_handler)
        self.ordinal_index_le.textEdited['QString'].connect(data_preprocessing.parameter_changed_handler)
        self.onehot_index_le.textEdited['QString'].connect(data_preprocessing.parameter_changed_handler)
        self.set_target_cb.clicked['bool'].connect(data_preprocessing.parameter_changed_handler)
        self.target_index_comb.currentTextChanged['QString'].connect(data_preprocessing.parameter_changed_handler)
        self.finish_btn.clicked.connect(data_preprocessing.finish_handler)
        self.apply_btn.clicked.connect(data_preprocessing.apply_handler)
        self.scaled_cb.clicked['bool'].connect(data_preprocessing.parameter_changed_handler)
        self.scale_by_comb.currentIndexChanged['int'].connect(data_preprocessing.parameter_changed_handler)
        QtCore.QMetaObject.connectSlotsByName(data_preprocessing)

    def retranslateUi(self, data_preprocessing):
        _translate = QtCore.QCoreApplication.translate
        data_preprocessing.setWindowTitle(_translate("data_preprocessing", "Data Preprocessing"))
        self.target_label_groupBox.setToolTip(_translate("data_preprocessing", "<html><head/><body><p>the class variable you want to predict</p></body></html>"))
        self.target_label_groupBox.setTitle(_translate("data_preprocessing", "Target Label"))
        self.set_target_cb.setText(_translate("data_preprocessing", "response variable at index of "))
        self.text_feature_groupBox.setTitle(_translate("data_preprocessing", "Text Feature"))
        self.drop_feature_by_index_cb.setText(_translate("data_preprocessing", "drop feature at column index of "))
        self.drop_index_le.setText(_translate("data_preprocessing", "0,482"))
        self.label_5.setText(_translate("data_preprocessing", "Feature index from 0"))
        self.trans_into_ordinal_cb.setText(_translate("data_preprocessing", "transform the text feature into ordinal"))
        self.trans_into_onehot_cb.setText(_translate("data_preprocessing", "transform the text feature into one-hot"))
        self.scale_groupBox.setTitle(_translate("data_preprocessing", "Scale Features ( Except the target feature )"))
        self.scaled_cb.setText(_translate("data_preprocessing", "features scaled by"))
        self.scale_by_comb.setItemText(0, _translate("data_preprocessing", "StandardScaler"))
        self.scale_by_comb.setItemText(1, _translate("data_preprocessing", "MinMaxScaler"))
        self.scale_by_comb.setItemText(2, _translate("data_preprocessing", "RobustScaler"))
        self.table_groupBox.setTitle(_translate("data_preprocessing", "Preview *unprocessed"))
        self.apply_btn.setText(_translate("data_preprocessing", "Apply"))
        self.finish_btn.setText(_translate("data_preprocessing", "Finish"))
        self.reset_btn.setText(_translate("data_preprocessing", "Reset Data From Previous"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    data_preprocessing = QtWidgets.QDialog()
    ui = Ui_data_preprocessing()
    ui.setupUi(data_preprocessing)
    data_preprocessing.show()
    sys.exit(app.exec_())
