# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'file_reader_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(415, 590)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 540, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.open_file_btn = Open_File_Btn(Dialog)
        self.open_file_btn.setGeometry(QtCore.QRect(40, 50, 111, 31))
        self.open_file_btn.setObjectName("open_file_btn")
        self.display_file_name_label = QtWidgets.QLabel(Dialog)
        self.display_file_name_label.setGeometry(QtCore.QRect(50, 110, 131, 41))
        self.display_file_name_label.setStyleSheet("font: 14pt \"Arial\";")
        self.display_file_name_label.setObjectName("display_file_name_label")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        self.open_file_btn.clicked.connect(Dialog.open_file_btn_clicked)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.open_file_btn.setText(_translate("Dialog", "Open File"))
        self.display_file_name_label.setText(_translate("Dialog", "No File Selcted"))
from open_file_btn import Open_File_Btn


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
