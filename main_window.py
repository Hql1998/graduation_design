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
        self.setup_ui_subtle()

        # widget management
        self.function_widget_dict ={"file_reader":[],
                                    "deal_empty": [],
                                    "data_preprocessing": [],
                                    "lasso_logistic_regression": [],
                                    "random_forest_classifier": [],
                                    "svm_classifier": [],
                                    "naive_bayes_classifier": [],
                                    "knn_classifier": []
                                    }

    def setup_ui_subtle(self):

        # self.tool_bar = QToolBar(self)
        # self.tool_bar.setMaximumHeight(20)
        # self.tool_bar.setFloatable(False)
        #
        # start_action = QAction(QIcon(QPixmap(":/main_window/start.png").scaled(20,20)), "start", self)
        # self.tool_bar.addAction(start_action)
        # stop_action = QAction(QIcon(QPixmap(":/main_window/stop.png").scaled(20, 20)), "stop", self)
        # self.tool_bar.addAction(stop_action)
        #
        # self.addToolBar(self.tool_bar)

        self.status_bar.showMessage("success to load on the mainwindow", 5000)

    def file_reader_btn_clicked_handler(self):
        frfw = File_Reader_Function_Widget(self.draw_scroll_area_content)
        frfw.icon_btn.setText("FReader")
        frfw.show()
        self.function_widget_dict[frfw.class_name].append(frfw)

    def deal_with_empty_btn_clicked_handler(self):
        defw = Deal_With_Empty_Value_Function_Widget(self.draw_scroll_area_content)
        defw.icon_btn.setText("Empty")
        defw.show()
        self.function_widget_dict[defw.class_name].append(defw)
    def drop_transform_btn_clicked_handler(self):
        dpfw = Data_Preprocessing_Function_Widget(self.draw_scroll_area_content)
        dpfw.icon_btn.setText("Preproce")
        dpfw.show()
        self.function_widget_dict[dpfw.class_name].append(dpfw)
    def lasso_logistic_regression_btn_clicked_handler(self):
        llrfw = Lasso_Logistic_Regression_Function_Widget(self.draw_scroll_area_content)
        llrfw.icon_btn.setText("LLRCV")
        llrfw.show()
        self.function_widget_dict[llrfw.class_name].append(llrfw)

    def random_forest_classifier_btn_clicked_handler(self):
        rfcfw = Random_Forest_Classifier_Function_Widget(self.draw_scroll_area_content)
        rfcfw.icon_btn.setText("RFCCV")
        rfcfw.show()
        self.function_widget_dict[rfcfw.class_name].append(rfcfw)

    def svm_classifier_btn_clicked_handler(self):
        svmcfw = SVM_Classifier_Function_Widget(self.draw_scroll_area_content)
        svmcfw.icon_btn.setText("SVMCV")
        svmcfw.show()
        self.function_widget_dict[svmcfw.class_name].append(svmcfw)

    def naive_bayes_classifier_btn_clicked_handler(self):
        nbcfw = Naive_Bayes_Classifier_Function_Widget(self.draw_scroll_area_content)
        nbcfw.icon_btn.setText("NB")
        nbcfw.show()
        self.function_widget_dict[nbcfw.class_name].append(nbcfw)

    def knn_classifier_btn_clicked_handler(self):
        knncfw = KNN_Classifier_Function_Widget(self.draw_scroll_area_content)
        knncfw.icon_btn.setText("KNNCV")
        knncfw.show()
        self.function_widget_dict[knncfw.class_name].append(knncfw)



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


