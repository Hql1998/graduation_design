from PyQt5.Qt import *
import sys
from global_imports import *

class Window(QMainWindow):
    def __init__(self, iniWindowWidth, iniWindowHeight, windowTitle):
        super().__init__()
        self.resize(iniWindowWidth, iniWindowHeight)
        self.setMinimumSize(400, 300)
        self.setWindowTitle(windowTitle)
        self.setup_ui(iniWindowWidth, iniWindowHeight)


    def setup_ui(self, iniWindowWidth, iniWindowHeight):
        # file_btn = OpenFileBtn(self)
        # file_btn.setText("File Reader")
        # file_btn.setFlat(True)
        # file_btn.setStyleSheet("background-color:gray")
        # file_btn.resize(mid_size_widget)

        OpenFileBtn.open_file(self)



        return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window(800, 800, "my_graduation_design")
    window.show()
    sys.exit(app.exec_())