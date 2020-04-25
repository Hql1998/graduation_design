import numpy as np
import pandas as pd
import matplotlib
import sklearn
from PyQt5.Qt import *
import sys

#Custom imports
import GUI_part.resouce_rc
from open_file_btn import *
from GUI_part.main_window_ui import Ui_MainWindow
from function_widget import *
from file_reader_function_widget import File_Reader_Function_Widget
from data_preprocessing_function_widget import Data_Preprocessing_Function_Widget
from deal_with_empty_value_function_widget import Deal_With_Empty_Value_Function_Widget
import pickle