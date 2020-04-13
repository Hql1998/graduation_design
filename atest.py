from PyQt5.Qt import *
import sys


def initial(window,window_title,width,height):
    screen_size = app.primaryScreen().size()
    window_size = QSize(width,height)
    center_point = QPoint((screen_size.width() - window_size.width()) / 2,
                          (screen_size.height() - window_size.height()) / 2)

    window.resize(window_size)
    window.move(center_point)
    window.setWindowTitle(window_title)


app = QApplication(sys.argv)

window = QWidget()
window_title = "good"
window.setObjectName("main window: ")
initial(window,window_title, 500, 500)

l1 = QLabel("good",window)
l1.move(100, 100)

l2 = QLabel("good",window)
l2.move(200, 200)

l3 = QLabel("good",window)
l3.move(300, 300)

print(l3.parent().findChildren(QLabel))


window.show()

sys.exit(app.exec_())