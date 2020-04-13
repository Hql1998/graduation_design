from global_imports import *


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent=None, *args, **kwargs)
        self.setAttribute(Qt.WA_StyledBackground, True)

        # load UI resources
        self.setupUi(self)

        self.status_bar = self.statusBar()
        self.setup_ui_subtle()

    def setup_ui_subtle(self):

        self.status_bar.showMessage("success to load on the mainwindow", 5000)
        # self.draw_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.fw = File_Reader_Function_Widget(self.draw_scroll_area)
        self.fw.icon_btn.setText("file reader")

        self.fw1 = Function_Widget(self.draw_scroll_area)
        self.fw1.icon_btn.setText("icon2")

        self.fw2 = Function_Widget(self.draw_scroll_area)
        self.fw2.icon_btn.setText("icon3")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())


