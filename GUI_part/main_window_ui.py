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
        MainWindow.resize(1104, 795)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        MainWindow.setDockNestingEnabled(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setMinimumSize(QtCore.QSize(900, 750))
        self.tabWidget.setStyleSheet("background-color: rgb(255, 170, 0);")
        self.tabWidget.setObjectName("tabWidget")
        self.draw_tab = QtWidgets.QWidget()
        self.draw_tab.setObjectName("draw_tab")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.draw_tab)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.draw_scroll_area = QtWidgets.QScrollArea(self.draw_tab)
        self.draw_scroll_area.setWidgetResizable(True)
        self.draw_scroll_area.setObjectName("draw_scroll_area")
        self.draw_scroll_area_content = QtWidgets.QWidget()
        self.draw_scroll_area_content.setGeometry(QtCore.QRect(0, 0, 874, 705))
        self.draw_scroll_area_content.setStyleSheet("background-image: url(:/draw_tab/draw_back_grid.png);")
        self.draw_scroll_area_content.setObjectName("draw_scroll_area_content")
        self.draw_scroll_area.setWidget(self.draw_scroll_area_content)
        self.horizontalLayout_4.addWidget(self.draw_scroll_area)
        self.tabWidget.addTab(self.draw_tab, "")
        self.log_tab = QtWidgets.QWidget()
        self.log_tab.setObjectName("log_tab")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.log_tab)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.log_scroll_tab = QtWidgets.QScrollArea(self.log_tab)
        self.log_scroll_tab.setStyleSheet("")
        self.log_scroll_tab.setWidgetResizable(True)
        self.log_scroll_tab.setObjectName("log_scroll_tab")
        self.log_scroll_area_content = QtWidgets.QWidget()
        self.log_scroll_area_content.setGeometry(QtCore.QRect(0, 0, 874, 705))
        self.log_scroll_area_content.setObjectName("log_scroll_area_content")
        self.log_scroll_tab.setWidget(self.log_scroll_area_content)
        self.horizontalLayout_5.addWidget(self.log_scroll_tab)
        self.tabWidget.addTab(self.log_tab, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1104, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menuedit = QtWidgets.QMenu(self.menubar)
        self.menuedit.setObjectName("menuedit")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget.setMinimumSize(QtCore.QSize(200, 143))
        self.dockWidget.setMaximumSize(QtCore.QSize(400, 524287))
        self.dockWidget.setStyleSheet("\n"
"background-color: rgb(255, 255, 127);")
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
        self.toolBox.setStyleSheet("background-color: rgb(255, 255, 0);")
        self.toolBox.setObjectName("toolBox")
        self.page = QtWidgets.QWidget()
        self.page.setGeometry(QtCore.QRect(0, 0, 200, 676))
        self.page.setObjectName("page")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.page)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.widget = QtWidgets.QWidget(self.page)
        self.widget.setObjectName("widget")
        self.horizontalLayout_2.addWidget(self.widget)
        self.toolBox.addItem(self.page, "")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setGeometry(QtCore.QRect(0, 0, 200, 676))
        self.page_2.setObjectName("page_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.page_2)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.widget_2 = QtWidgets.QWidget(self.page_2)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_3.addWidget(self.widget_2)
        self.toolBox.addItem(self.page_2, "")
        self.verticalLayout.addWidget(self.toolBox)
        self.dockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget)
        self.actionopen_file = QtWidgets.QAction(MainWindow)
        self.actionopen_file.setObjectName("actionopen_file")
        self.menu.addAction(self.actionopen_file)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menuedit.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.toolBox.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.draw_tab), _translate("MainWindow", "Experiment Area"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.log_tab), _translate("MainWindow", "Logs"))
        self.menu.setTitle(_translate("MainWindow", "files"))
        self.menuedit.setTitle(_translate("MainWindow", "edit"))
        self.dockWidget.setWindowTitle(_translate("MainWindow", "Function Gadget"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page), _translate("MainWindow", "Frequent Use"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), _translate("MainWindow", "Standard Library"))
        self.actionopen_file.setText(_translate("MainWindow", "open file"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
