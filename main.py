from main_window import *
from global_imports import *


import sys
app = QApplication(sys.argv)
window = Window()
# window.state.selected_widgets

window.show()
sys.exit(app.exec_())