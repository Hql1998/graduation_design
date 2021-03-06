# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'random_forest_classifier.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("self")
        Dialog.resize(464, 858)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_1 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_1.setObjectName("groupBox_1")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_1)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_1)
        self.groupBox.setEnabled(True)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.random_search_cb = QtWidgets.QCheckBox(self.groupBox)
        self.random_search_cb.setChecked(True)
        self.random_search_cb.setObjectName("random_search_cb")
        self.gridLayout.addWidget(self.random_search_cb, 0, 0, 1, 1)
        self.grid_search_cb = QtWidgets.QCheckBox(self.groupBox)
        self.grid_search_cb.setObjectName("grid_search_cb")
        self.gridLayout.addWidget(self.grid_search_cb, 0, 2, 1, 3)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setToolTip("")
        self.label.setStatusTip("")
        self.label.setStyleSheet("font-size:12px;")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 2)
        self.cv_folds_sp = QtWidgets.QSpinBox(self.groupBox)
        self.cv_folds_sp.setMaximum(10)
        self.cv_folds_sp.setProperty("value", 5)
        self.cv_folds_sp.setObjectName("cv_folds_sp")
        self.gridLayout.addWidget(self.cv_folds_sp, 1, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setStyleSheet("font-size:12px;")
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.tree_num_start_sp = QtWidgets.QSpinBox(self.groupBox)
        self.tree_num_start_sp.setMinimum(20)
        self.tree_num_start_sp.setMaximum(500)
        self.tree_num_start_sp.setSingleStep(50)
        self.tree_num_start_sp.setProperty("value", 20)
        self.tree_num_start_sp.setObjectName("tree_num_start_sp")
        self.gridLayout.addWidget(self.tree_num_start_sp, 2, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setStyleSheet("font-size:12px;")
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 3, 1, 1)
        self.tree_num_sp = QtWidgets.QSpinBox(self.groupBox)
        self.tree_num_sp.setMinimum(2)
        self.tree_num_sp.setMaximum(100)
        self.tree_num_sp.setSingleStep(5)
        self.tree_num_sp.setProperty("value", 20)
        self.tree_num_sp.setObjectName("tree_num_sp")
        self.gridLayout.addWidget(self.tree_num_sp, 2, 8, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setStyleSheet("font-size:12px;")
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 3, 3, 1, 1)
        self.feature_prop_num_sp = QtWidgets.QSpinBox(self.groupBox)
        self.feature_prop_num_sp.setMinimum(2)
        self.feature_prop_num_sp.setMaximum(100)
        self.feature_prop_num_sp.setSingleStep(5)
        self.feature_prop_num_sp.setProperty("value", 5)
        self.feature_prop_num_sp.setObjectName("feature_prop_num_sp")
        self.gridLayout.addWidget(self.feature_prop_num_sp, 3, 8, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)
        self.max_depth_start_sp = QtWidgets.QSpinBox(self.groupBox)
        self.max_depth_start_sp.setToolTip("")
        self.max_depth_start_sp.setStatusTip("")
        self.max_depth_start_sp.setMinimum(2)
        self.max_depth_start_sp.setMaximum(200)
        self.max_depth_start_sp.setSingleStep(10)
        self.max_depth_start_sp.setProperty("value", 3)
        self.max_depth_start_sp.setObjectName("max_depth_start_sp")
        self.gridLayout.addWidget(self.max_depth_start_sp, 4, 2, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.groupBox)
        self.label_20.setStyleSheet("font-size:12px;")
        self.label_20.setObjectName("label_20")
        self.gridLayout.addWidget(self.label_20, 4, 3, 1, 1)
        self.max_depth_end_sp = QtWidgets.QSpinBox(self.groupBox)
        self.max_depth_end_sp.setMinimum(10)
        self.max_depth_end_sp.setMaximum(200)
        self.max_depth_end_sp.setSingleStep(10)
        self.max_depth_end_sp.setProperty("value", 10)
        self.max_depth_end_sp.setObjectName("max_depth_end_sp")
        self.gridLayout.addWidget(self.max_depth_end_sp, 4, 4, 1, 2)
        self.label_19 = QtWidgets.QLabel(self.groupBox)
        self.label_19.setStyleSheet("font-size:12px;")
        self.label_19.setObjectName("label_19")
        self.gridLayout.addWidget(self.label_19, 4, 6, 1, 2)
        self.max_depth_num_sp = QtWidgets.QSpinBox(self.groupBox)
        self.max_depth_num_sp.setMinimum(2)
        self.max_depth_num_sp.setMaximum(100)
        self.max_depth_num_sp.setSingleStep(5)
        self.max_depth_num_sp.setProperty("value", 5)
        self.max_depth_num_sp.setObjectName("max_depth_num_sp")
        self.gridLayout.addWidget(self.max_depth_num_sp, 4, 8, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.groupBox)
        self.label_21.setObjectName("label_21")
        self.gridLayout.addWidget(self.label_21, 5, 0, 1, 1)
        self.min_split_start_sp = QtWidgets.QSpinBox(self.groupBox)
        self.min_split_start_sp.setToolTip("")
        self.min_split_start_sp.setStatusTip("")
        self.min_split_start_sp.setMinimum(2)
        self.min_split_start_sp.setMaximum(200)
        self.min_split_start_sp.setSingleStep(10)
        self.min_split_start_sp.setProperty("value", 2)
        self.min_split_start_sp.setObjectName("min_split_start_sp")
        self.gridLayout.addWidget(self.min_split_start_sp, 5, 2, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.groupBox)
        self.label_22.setStyleSheet("font-size:12px;")
        self.label_22.setObjectName("label_22")
        self.gridLayout.addWidget(self.label_22, 5, 3, 1, 1)
        self.min_split_end_sp = QtWidgets.QSpinBox(self.groupBox)
        self.min_split_end_sp.setMinimum(10)
        self.min_split_end_sp.setMaximum(200)
        self.min_split_end_sp.setSingleStep(10)
        self.min_split_end_sp.setProperty("value", 20)
        self.min_split_end_sp.setObjectName("min_split_end_sp")
        self.gridLayout.addWidget(self.min_split_end_sp, 5, 4, 1, 2)
        self.label_23 = QtWidgets.QLabel(self.groupBox)
        self.label_23.setStyleSheet("font-size:12px;")
        self.label_23.setObjectName("label_23")
        self.gridLayout.addWidget(self.label_23, 5, 6, 1, 2)
        self.min_split_num_sp = QtWidgets.QSpinBox(self.groupBox)
        self.min_split_num_sp.setMinimum(2)
        self.min_split_num_sp.setMaximum(100)
        self.min_split_num_sp.setSingleStep(5)
        self.min_split_num_sp.setProperty("value", 5)
        self.min_split_num_sp.setObjectName("min_split_num_sp")
        self.gridLayout.addWidget(self.min_split_num_sp, 5, 8, 1, 1)
        self.label_38 = QtWidgets.QLabel(self.groupBox)
        self.label_38.setObjectName("label_38")
        self.gridLayout.addWidget(self.label_38, 6, 0, 1, 1)
        self.min_leaf_start_sp = QtWidgets.QSpinBox(self.groupBox)
        self.min_leaf_start_sp.setToolTip("")
        self.min_leaf_start_sp.setStatusTip("")
        self.min_leaf_start_sp.setMinimum(2)
        self.min_leaf_start_sp.setMaximum(200)
        self.min_leaf_start_sp.setSingleStep(10)
        self.min_leaf_start_sp.setProperty("value", 2)
        self.min_leaf_start_sp.setObjectName("min_leaf_start_sp")
        self.gridLayout.addWidget(self.min_leaf_start_sp, 6, 2, 1, 1)
        self.label_40 = QtWidgets.QLabel(self.groupBox)
        self.label_40.setStyleSheet("font-size:12px;")
        self.label_40.setObjectName("label_40")
        self.gridLayout.addWidget(self.label_40, 6, 3, 1, 1)
        self.min_leaf_end_sp = QtWidgets.QSpinBox(self.groupBox)
        self.min_leaf_end_sp.setMinimum(10)
        self.min_leaf_end_sp.setMaximum(200)
        self.min_leaf_end_sp.setSingleStep(10)
        self.min_leaf_end_sp.setProperty("value", 20)
        self.min_leaf_end_sp.setObjectName("min_leaf_end_sp")
        self.gridLayout.addWidget(self.min_leaf_end_sp, 6, 4, 1, 2)
        self.label_39 = QtWidgets.QLabel(self.groupBox)
        self.label_39.setStyleSheet("font-size:12px;")
        self.label_39.setObjectName("label_39")
        self.gridLayout.addWidget(self.label_39, 6, 6, 1, 2)
        self.min_leaf_num_sp = QtWidgets.QSpinBox(self.groupBox)
        self.min_leaf_num_sp.setMinimum(2)
        self.min_leaf_num_sp.setMaximum(100)
        self.min_leaf_num_sp.setSingleStep(5)
        self.min_leaf_num_sp.setProperty("value", 5)
        self.min_leaf_num_sp.setObjectName("min_leaf_num_sp")
        self.gridLayout.addWidget(self.min_leaf_num_sp, 6, 8, 1, 1)
        self.balanced_class_weight_cb = QtWidgets.QCheckBox(self.groupBox)
        self.balanced_class_weight_cb.setObjectName("balanced_class_weight_cb")
        self.gridLayout.addWidget(self.balanced_class_weight_cb, 7, 0, 1, 2)
        self.label_10 = QtWidgets.QLabel(self.groupBox)
        self.label_10.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 8, 0, 1, 2)
        self.scoring_comb = QtWidgets.QComboBox(self.groupBox)
        self.scoring_comb.setObjectName("scoring_comb")
        self.scoring_comb.addItem("")
        self.scoring_comb.setItemText(0, "")
        self.gridLayout.addWidget(self.scoring_comb, 8, 2, 1, 1)
        self.feature_prop_start_dsp = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.feature_prop_start_dsp.setMinimum(0.01)
        self.feature_prop_start_dsp.setMaximum(0.9)
        self.feature_prop_start_dsp.setSingleStep(0.1)
        self.feature_prop_start_dsp.setObjectName("feature_prop_start_dsp")
        self.gridLayout.addWidget(self.feature_prop_start_dsp, 3, 2, 1, 1)
        self.feature_prop_end_dsp = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.feature_prop_end_dsp.setMinimum(0.01)
        self.feature_prop_end_dsp.setMaximum(1.0)
        self.feature_prop_end_dsp.setSingleStep(0.1)
        self.feature_prop_end_dsp.setProperty("value", 0.5)
        self.feature_prop_end_dsp.setObjectName("feature_prop_end_dsp")
        self.gridLayout.addWidget(self.feature_prop_end_dsp, 3, 4, 1, 2)
        self.tree_end_end_sp = QtWidgets.QSpinBox(self.groupBox)
        self.tree_end_end_sp.setMinimum(100)
        self.tree_end_end_sp.setMaximum(500)
        self.tree_end_end_sp.setSingleStep(50)
        self.tree_end_end_sp.setProperty("value", 100)
        self.tree_end_end_sp.setObjectName("tree_end_end_sp")
        self.gridLayout.addWidget(self.tree_end_end_sp, 2, 4, 1, 2)
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setStyleSheet("font-size:12px;")
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 6, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setStyleSheet("font-size:12px;")
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 3, 6, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_1)
        self.groupBox_4.setObjectName("groupBox_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_4)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.load_model_cb = QtWidgets.QCheckBox(self.groupBox_4)
        self.load_model_cb.setEnabled(True)
        self.load_model_cb.setObjectName("load_model_cb")
        self.horizontalLayout_2.addWidget(self.load_model_cb)
        self.display_model_name_label = QtWidgets.QLabel(self.groupBox_4)
        self.display_model_name_label.setStyleSheet("")
        self.display_model_name_label.setObjectName("display_model_name_label")
        self.horizontalLayout_2.addWidget(self.display_model_name_label, 0, QtCore.Qt.AlignVCenter)
        self.open_model_btn = Open_Model_Btn(self.groupBox_4)
        self.open_model_btn.setEnabled(False)
        self.open_model_btn.setObjectName("open_model_btn")
        self.horizontalLayout_2.addWidget(self.open_model_btn)
        self.verticalLayout_3.addWidget(self.groupBox_4)
        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout.addWidget(self.groupBox_1)
        self.groupBox_2 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.plot_roc_cb = QtWidgets.QCheckBox(self.groupBox_2)
        self.plot_roc_cb.setChecked(True)
        self.plot_roc_cb.setObjectName("plot_roc_cb")
        self.gridLayout_2.addWidget(self.plot_roc_cb, 0, 0, 1, 1)
        self.output_cla_rep_cb = QtWidgets.QCheckBox(self.groupBox_2)
        self.output_cla_rep_cb.setChecked(True)
        self.output_cla_rep_cb.setObjectName("output_cla_rep_cb")
        self.gridLayout_2.addWidget(self.output_cla_rep_cb, 1, 0, 1, 1)
        self.output_confusion_cb = QtWidgets.QCheckBox(self.groupBox_2)
        self.output_confusion_cb.setChecked(True)
        self.output_confusion_cb.setObjectName("output_confusion_cb")
        self.gridLayout_2.addWidget(self.output_confusion_cb, 1, 1, 1, 2)
        self.feature_filter_cb = QtWidgets.QCheckBox(self.groupBox_2)
        self.feature_filter_cb.setObjectName("feature_filter_cb")
        self.gridLayout_2.addWidget(self.feature_filter_cb, 2, 0, 1, 2)
        self.save_file_cb = QtWidgets.QCheckBox(self.groupBox_2)
        self.save_file_cb.setObjectName("save_file_cb")
        self.gridLayout_2.addWidget(self.save_file_cb, 3, 0, 1, 2)
        self.save_file_btn = Save_File_Btn(self.groupBox_2)
        self.save_file_btn.setEnabled(False)
        self.save_file_btn.setObjectName("save_file_btn")
        self.gridLayout_2.addWidget(self.save_file_btn, 4, 0, 1, 1)
        self.save_file_label = QtWidgets.QLabel(self.groupBox_2)
        self.save_file_label.setObjectName("save_file_label")
        self.gridLayout_2.addWidget(self.save_file_label, 4, 1, 1, 2, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.save_model_cb = QtWidgets.QCheckBox(self.groupBox_2)
        self.save_model_cb.setObjectName("save_model_cb")
        self.gridLayout_2.addWidget(self.save_model_cb, 5, 0, 1, 1)
        self.save_model_btn = Save_Model_Btn(self.groupBox_2)
        self.save_model_btn.setEnabled(False)
        self.save_model_btn.setObjectName("save_model_btn")
        self.gridLayout_2.addWidget(self.save_model_btn, 6, 0, 1, 1)
        self.save_model_label = QtWidgets.QLabel(self.groupBox_2)
        self.save_model_label.setObjectName("save_model_label")
        self.gridLayout_2.addWidget(self.save_model_label, 6, 1, 1, 2, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.feature_importance_le = QtWidgets.QLineEdit(self.groupBox_2)
        self.feature_importance_le.setMinimumSize(QtCore.QSize(50, 0))
        self.feature_importance_le.setMaximumSize(QtCore.QSize(100, 16777215))
        self.feature_importance_le.setObjectName("feature_importance_le")
        self.gridLayout_2.addWidget(self.feature_importance_le, 2, 2, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox_3)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_2.addWidget(self.textBrowser)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setMinimumSize(QtCore.QSize(0, 20))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.apply_btn = QtWidgets.QPushButton(self.widget)
        self.apply_btn.setObjectName("apply_btn")
        self.horizontalLayout.addWidget(self.apply_btn)
        self.reset_btn = QtWidgets.QPushButton(self.widget)
        self.reset_btn.setObjectName("reset_btn")
        self.horizontalLayout.addWidget(self.reset_btn)
        self.finish_btn = QtWidgets.QPushButton(self.widget)
        self.finish_btn.setObjectName("finish_btn")
        self.horizontalLayout.addWidget(self.finish_btn)
        self.verticalLayout.addWidget(self.widget)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)

        self.retranslateUi(Dialog)
        self.load_model_cb.toggled['bool'].connect(Dialog.load_model_cb_toggled_handler)
        self.open_model_btn.clicked.connect(Dialog.open_model_btn_clicked_handler)
        self.save_file_cb.toggled['bool'].connect(Dialog.save_file_cb_toggled_handler)
        self.save_file_btn.clicked.connect(Dialog.save_file_btn_clicked_handler)
        self.save_model_cb.toggled['bool'].connect(Dialog.save_model_cb_toggled_handler)
        self.save_model_btn.clicked.connect(Dialog.save_model_btn_clicked_handler)
        self.apply_btn.clicked.connect(Dialog.apply_handler)
        self.finish_btn.clicked.connect(Dialog.finish_handler)
        self.feature_filter_cb.toggled['bool'].connect(Dialog.feature_filter_toggled_handler)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("self", "Random Forest Classifier"))
        self.groupBox_1.setTitle(_translate("self", "parameter setting"))
        self.groupBox.setTitle(_translate("self", "train a model"))
        self.random_search_cb.setText(_translate("self", "RandomizedSearchCV"))
        self.grid_search_cb.setText(_translate("self", "GridSearchCV"))
        self.label.setText(_translate("self", "Folds for Cross Validatin"))
        self.label_2.setToolTip(_translate("self", "Number of trees in random forest"))
        self.label_2.setStatusTip(_translate("self", "Number of trees in random forest"))
        self.label_2.setText(_translate("self", "Number of Trees From"))
        self.label_3.setText(_translate("self", "to"))
        self.label_5.setToolTip(_translate("self", "proportion of all features to consider at every split"))
        self.label_5.setStatusTip(_translate("self", "proportion of all features to consider at every split"))
        self.label_5.setText(_translate("self", "Proportion of All Features"))
        self.label_7.setText(_translate("self", "to"))
        self.label_6.setToolTip(_translate("self", "Maximum number of levels in tree"))
        self.label_6.setStatusTip(_translate("self", "Maximum number of levels in tree"))
        self.label_6.setText(_translate("self", "Max Depth From"))
        self.label_20.setText(_translate("self", "to"))
        self.label_19.setText(_translate("self", "select"))
        self.label_21.setToolTip(_translate("self", "Maximum number of levels in tree"))
        self.label_21.setStatusTip(_translate("self", "Maximum number of levels in tree"))
        self.label_21.setText(_translate("self", "Min Samples Split From"))
        self.label_22.setText(_translate("self", "to"))
        self.label_23.setText(_translate("self", "select"))
        self.label_38.setToolTip(_translate("self", "Maximum number of levels in tree"))
        self.label_38.setStatusTip(_translate("self", "Maximum number of levels in tree"))
        self.label_38.setText(_translate("self", "Min Samples Leaf From"))
        self.label_40.setText(_translate("self", "to"))
        self.label_39.setText(_translate("self", "select"))
        self.balanced_class_weight_cb.setText(_translate("self", "balanced class weight"))
        self.label_10.setText(_translate("self", "CV target scoring metrix"))
        self.label_4.setText(_translate("self", "select"))
        self.label_9.setText(_translate("self", "select"))
        self.groupBox_4.setTitle(_translate("self", "Or load a model"))
        self.load_model_cb.setText(_translate("self", "Load Model"))
        self.display_model_name_label.setText(_translate("self", "No File Selcted"))
        self.open_model_btn.setText(_translate("self", "Load model(joblib file)"))
        self.groupBox_2.setTitle(_translate("self", "Out Parameters"))
        self.plot_roc_cb.setText(_translate("self", "Plot ROC on training and testing datsets"))
        self.output_cla_rep_cb.setText(_translate("self", "Output classification report"))
        self.output_confusion_cb.setText(_translate("self", "Output confusion matrix"))
        self.feature_filter_cb.setText(_translate("self", "Only keep features with importance greater than "))
        self.save_file_cb.setText(_translate("self", "Save (transformed) training (and testng) file(s) to"))
        self.save_file_btn.setText(_translate("self", "Save the Files"))
        self.save_file_label.setText(_translate("self", "No Directory selected"))
        self.save_model_cb.setText(_translate("self", "Save fitted model into"))
        self.save_model_btn.setText(_translate("self", "Save the Model"))
        self.save_model_label.setText(_translate("self", "No Directory selected"))
        self.groupBox_3.setTitle(_translate("self", "Output field"))
        self.apply_btn.setText(_translate("self", "Apply"))
        self.reset_btn.setText(_translate("self", "Reset From Previous"))
        self.finish_btn.setText(_translate("self", "Finish"))
from open_model_btn import Open_Model_Btn
from save_file_btn import Save_File_Btn
from save_model_btn import Save_Model_Btn


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
