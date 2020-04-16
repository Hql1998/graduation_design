from PyQt5.Qt import *


class Auxiliary_Table(QTableWidget):

    def select_column(self,row_index,column_index):
        self.selectColumn(column_index)


