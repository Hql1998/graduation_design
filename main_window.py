from global_imports import *


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent=None, *args, **kwargs)
        self.setAttribute(Qt.WA_StyledBackground, True)

        # status management
        self.status_initiate()

        # load UI resources
        self.setupUi(self)
        self.status_bar = self.statusBar()
        self.tool_bar = QToolBar(self)
        self.setup_ui_subtle()

    def setup_ui_subtle(self):

        self.tool_bar.setMaximumHeight(20)
        self.tool_bar.setFloatable(False)

        start_action = QAction(QIcon(QPixmap(":/main_window/start.png").scaled(20,20)), "start", self)
        self.tool_bar.addAction(start_action)
        stop_action = QAction(QIcon(QPixmap(":/main_window/stop.png").scaled(20, 20)), "stop", self)
        self.tool_bar.addAction(stop_action)

        self.addToolBar(self.tool_bar)

        # self.draw_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.fw = File_Reader_Function_Widget(self.draw_scroll_area_content)
        self.fw.icon_btn.setText("file reader")

        self.fw1 = Data_Preprocessing_Function_Widget(self.draw_scroll_area_content)
        self.fw1.icon_btn.setText("Data_Preproce")

        self.fw2 = Function_Widget(self.draw_scroll_area_content)
        self.fw2.icon_btn.setText("icon3")

        self.status_bar.showMessage("success to load on the mainwindow", 5000)

    def status_initiate(self):
        if False:
            "判断文件里面有没有数据，如果有就加载"
        else:
            self.activeWidget = {"file_reader": [], "data_preprocessing": [], "feature_preprocessing": [],
                             "data_analysis": []}


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())


