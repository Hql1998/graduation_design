from PyQt5.Qt import *


class Split_Method_Combo(QComboBox):

    def target_handler(self,target_bool):
        if target_bool:
            self.clear()
            self.addItems(["Random Split", "Stratified Shuffle Split"])
        else:
            self.clear()
            self.addItems(["Random Split"])