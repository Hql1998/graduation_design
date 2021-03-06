# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1208, 828)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        MainWindow.setDockNestingEnabled(True)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setMinimumSize(QtCore.QSize(906, 765))
        self.tabWidget.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.tabWidget.setObjectName("tabWidget")
        self.draw_tab = QtWidgets.QWidget()
        self.draw_tab.setObjectName("draw_tab")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.draw_tab)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.draw_scroll_area = QtWidgets.QScrollArea(self.draw_tab)
        self.draw_scroll_area.setStyleSheet("")
        self.draw_scroll_area.setWidgetResizable(True)
        self.draw_scroll_area.setObjectName("draw_scroll_area")
        self.draw_scroll_area_content = Draw_Scroll_Area_Content()
        self.draw_scroll_area_content.setGeometry(QtCore.QRect(0, 0, 880, 743))
        self.draw_scroll_area_content.setStyleSheet("background-image: url(:/draw_tab/draw_back_grid.png);")
        self.draw_scroll_area_content.setObjectName("draw_scroll_area_content")
        self.draw_scroll_area.setWidget(self.draw_scroll_area_content)
        self.horizontalLayout_4.addWidget(self.draw_scroll_area)
        self.tabWidget.addTab(self.draw_tab, "")
        self.log_tab = QtWidgets.QWidget()
        self.log_tab.setObjectName("log_tab")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.log_tab)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.log_scroll_tab = QtWidgets.QScrollArea(self.log_tab)
        self.log_scroll_tab.setStyleSheet("")
        self.log_scroll_tab.setWidgetResizable(True)
        self.log_scroll_tab.setObjectName("log_scroll_tab")
        self.log_scroll_area_content = QtWidgets.QWidget()
        self.log_scroll_area_content.setGeometry(QtCore.QRect(0, 0, 89, 89))
        self.log_scroll_area_content.setObjectName("log_scroll_area_content")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.log_scroll_area_content)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.log_te = QtWidgets.QTextEdit(self.log_scroll_area_content)
        self.log_te.setReadOnly(True)
        self.log_te.setObjectName("log_te")
        self.verticalLayout_3.addWidget(self.log_te)
        self.log_scroll_tab.setWidget(self.log_scroll_area_content)
        self.verticalLayout_4.addWidget(self.log_scroll_tab)
        self.tabWidget.addTab(self.log_tab, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget.setMinimumSize(QtCore.QSize(280, 143))
        self.dockWidget.setMaximumSize(QtCore.QSize(500, 524287))
        self.dockWidget.setStyleSheet("background-color: rgba(159, 211, 167, 120);")
        self.dockWidget.setFloating(False)
        self.dockWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        self.dockWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.toolBox = QtWidgets.QToolBox(self.dockWidgetContents)
        self.toolBox.setMinimumSize(QtCore.QSize(200, 0))
        self.toolBox.setStyleSheet("background-color: rgba(159, 211, 167, 20);")
        self.toolBox.setObjectName("toolBox")
        self.page = QtWidgets.QWidget()
        self.page.setGeometry(QtCore.QRect(0, 0, 280, 732))
        self.page.setStyleSheet("")
        self.page.setObjectName("page")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.page)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.page)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_2.addWidget(self.pushButton_4)
        self.pushButton_3 = QtWidgets.QPushButton(self.page)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_2.addWidget(self.pushButton_3)
        self.pushButton = QtWidgets.QPushButton(self.page)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.page)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_5 = QtWidgets.QPushButton(self.page)
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout_2.addWidget(self.pushButton_5)
        self.pushButton_6 = QtWidgets.QPushButton(self.page)
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout_2.addWidget(self.pushButton_6)
        self.pushButton_7 = QtWidgets.QPushButton(self.page)
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout_2.addWidget(self.pushButton_7)
        self.pushButton_8 = QtWidgets.QPushButton(self.page)
        self.pushButton_8.setObjectName("pushButton_8")
        self.verticalLayout_2.addWidget(self.pushButton_8)
        self.toolBox.addItem(self.page, "")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setGeometry(QtCore.QRect(0, 0, 100, 30))
        self.page_2.setObjectName("page_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.page_2)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.toolBox.addItem(self.page_2, "")
        self.verticalLayout.addWidget(self.toolBox)
        self.dockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget)
        self.actionopen_file = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/main_window/add_new.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionopen_file.setIcon(icon)
        self.actionopen_file.setObjectName("actionopen_file")

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.toolBox.setCurrentIndex(0)
        self.pushButton_4.clicked.connect(MainWindow.file_reader_btn_clicked_handler)
        self.pushButton_3.clicked.connect(MainWindow.deal_with_empty_btn_clicked_handler)
        self.pushButton.clicked.connect(MainWindow.drop_transform_btn_clicked_handler)
        self.pushButton_2.clicked.connect(MainWindow.lasso_logistic_regression_btn_clicked_handler)
        self.pushButton_5.clicked.connect(MainWindow.random_forest_classifier_btn_clicked_handler)
        self.pushButton_6.clicked.connect(MainWindow.svm_classifier_btn_clicked_handler)
        self.pushButton_7.clicked.connect(MainWindow.naive_bayes_classifier_btn_clicked_handler)
        self.pushButton_8.clicked.connect(MainWindow.knn_classifier_btn_clicked_handler)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.draw_tab), _translate("MainWindow", "Experiment Area"))
        self.log_te.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; font-weight:600;\">Log Record:</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:16pt; font-weight:600;\"><br /></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.log_tab), _translate("MainWindow", "Logs"))
        self.dockWidget.setWindowTitle(_translate("MainWindow", "Function Gadget"))
        self.pushButton_4.setText(_translate("MainWindow", "File Reader"))
        self.pushButton_3.setText(_translate("MainWindow", "Deal with Empty Value"))
        self.pushButton.setText(_translate("MainWindow", "Drop & Transform & Standarization"))
        self.pushButton_2.setText(_translate("MainWindow", "Lasso Logistic Regression with CV"))
        self.pushButton_5.setText(_translate("MainWindow", "Random Forest Classifier with CV"))
        self.pushButton_6.setText(_translate("MainWindow", "SVM Classifier with CV"))
        self.pushButton_7.setText(_translate("MainWindow", "Naive Bayes Classifier"))
        self.pushButton_8.setText(_translate("MainWindow", "K Near Neighbors"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page), _translate("MainWindow", "Frequent Use"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), _translate("MainWindow", "Standard Library"))
        self.actionopen_file.setText(_translate("MainWindow", "open file"))
from draw_scroll_area_content import Draw_Scroll_Area_Content

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
