from global_imports import *


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent=None, *args, **kwargs)
        self.setAttribute(Qt.WA_StyledBackground, True)
        qApp.main_window = self

        # status management
        self.status_initiate()
        self.destroyed.connect(self.pickle_dump)

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
        self.frfw = File_Reader_Function_Widget(self.draw_scroll_area_content)
        self.frfw.icon_btn.setText("FReader")

        self.defw = Deal_With_Empty_Value_Function_Widget(self.draw_scroll_area_content)
        self.defw.icon_btn.setText("Empty")

        self.dpfw = Data_Preprocessing_Function_Widget(self.draw_scroll_area_content)
        self.dpfw.icon_btn.setText("Preproce")

        self.llrfw = Lasso_Logistic_Regression_Function_Widget(self.draw_scroll_area_content)
        self.llrfw.icon_btn.setText("LLRCV")

        self.status_bar.showMessage("success to load on the mainwindow", 5000)

    def status_initiate(self):
        if os.path.isfile('./temp/p.txt'):
        # "判断文件里面有没有数据，如果有就加载"
            f = open('./temp/p.txt', 'r')
            pickle.load(f).show()
            f.close()
        else:
            self.activeWidget = {"file_reader": [], "data_preprocessing": [], "feature_preprocessing": [],
                             "data_analysis": []}

    def pickle_dump(self):
        print("main_window destroyed")
        f = open('./temp/p.txt', 'w')
        pickle.dump(self, f)
        f.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())


