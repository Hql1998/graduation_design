from main_window import *


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Window()
    # window.state.selected_widgets

    window.show()
    sys.exit(app.exec_())