from PyQt5.Qt import *

class Lights_Change:
    @staticmethod
    def unprepared(light_label):
        light_label.setStyleSheet("""
        border-radius:9px;
        background: gray;
        """)
        light_label.status = "unprepared"
    @staticmethod
    def start(light_label):
        light_label.setStyleSheet("""
        border-radius:9px;
        background: rgb(85,217,132);
        """)
        light_label.status = "start"
    @staticmethod
    def processing(light_label):
        light_label.setStyleSheet("""
        border-radius:9px;
        background: yellow;
        """)
        light_label.status = "processing"
    @staticmethod
    def finish(light_label):
        light_label.setStyleSheet("""
        border-radius:9px;
        background: red;
        """)
        light_label.status = "finish"


class Light_Label(QLabel):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.status = "unprepared"
        self.setup_ui()

    def setup_ui(self):
        self.setFixedSize(18,18)
        self.setText("")
        Lights_Change.unprepared(self)