# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'file_reader_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_file_reader_dialog(object):
    def setupUi(self, file_reader_dialog):
        file_reader_dialog.setObjectName("file_reader_dialog")
        file_reader_dialog.resize(518, 753)
        file_reader_dialog.setStyleSheet("background: None;")
        self.verticalLayout = QtWidgets.QVBoxLayout(file_reader_dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.groupBox = QtWidgets.QGroupBox(file_reader_dialog)
        self.groupBox.setMinimumSize(QtCore.QSize(400, 100))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setContentsMargins(15, 15, 15, 15)
        self.gridLayout.setSpacing(15)
        self.gridLayout.setObjectName("gridLayout")
        self.open_file_btn = Open_File_Btn(self.groupBox)
        self.open_file_btn.setObjectName("open_file_btn")
        self.gridLayout.addWidget(self.open_file_btn, 0, 0, 1, 1)
        self.display_file_name_label = QtWidgets.QLabel(self.groupBox)
        self.display_file_name_label.setStyleSheet("")
        self.display_file_name_label.setObjectName("display_file_name_label")
        self.gridLayout.addWidget(self.display_file_name_label, 1, 1, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setStyleSheet("font: 12pt \"Arial\";")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox.setChecked(True)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout.addWidget(self.checkBox, 0, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 3)
        self.gridLayout.setRowStretch(0, 2)
        self.gridLayout.setRowStretch(1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        spacerItem1 = QtWidgets.QSpacerItem(20, 16, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.groupBox_2 = QtWidgets.QGroupBox(file_reader_dialog)
        self.groupBox_2.setMinimumSize(QtCore.QSize(500, 300))
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tableWidget = QtWidgets.QTableWidget(self.groupBox_2)
        self.tableWidget.setMinimumSize(QtCore.QSize(400, 300))
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.verticalLayout_2.addWidget(self.tableWidget)
        self.tableWidget_describe = Auxiliary_Table(self.groupBox_2)
        self.tableWidget_describe.setObjectName("tableWidget_describe")
        self.tableWidget_describe.setColumnCount(0)
        self.tableWidget_describe.setRowCount(0)
        self.verticalLayout_2.addWidget(self.tableWidget_describe)
        self.verticalLayout.addWidget(self.groupBox_2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.buttonBox = QtWidgets.QDialogButtonBox(file_reader_dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem3)

        self.retranslateUi(file_reader_dialog)
        self.buttonBox.accepted.connect(file_reader_dialog.accept)
        self.buttonBox.rejected.connect(file_reader_dialog.reject)
        self.open_file_btn.file_path_changed.connect(file_reader_dialog.open_file_btn_clicked)
        self.tableWidget.cellClicked['int','int'].connect(self.tableWidget_describe.select_column)
        self.tableWidget.cellChanged['int','int'].connect(self.tableWidget_describe.select_column)
        QtCore.QMetaObject.connectSlotsByName(file_reader_dialog)

    def retranslateUi(self, file_reader_dialog):
        _translate = QtCore.QCoreApplication.translate
        file_reader_dialog.setWindowTitle(_translate("file_reader_dialog", "File Reader Dialog"))
        self.groupBox.setTitle(_translate("file_reader_dialog", "Files"))
        self.open_file_btn.setText(_translate("file_reader_dialog", "Open File"))
        self.display_file_name_label.setText(_translate("file_reader_dialog", "No File Selcted"))
        self.label.setText(_translate("file_reader_dialog", "File Name: "))
        self.checkBox.setText(_translate("file_reader_dialog", "Column Name at First Line"))
        self.groupBox_2.setTitle(_translate("file_reader_dialog", "Preview"))
from auxiliary_table import Auxiliary_Table
from open_file_btn import Open_File_Btn


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    file_reader_dialog = QtWidgets.QDialog()
    ui = Ui_file_reader_dialog()
    ui.setupUi(file_reader_dialog)
    file_reader_dialog.show()
    sys.exit(app.exec_())
