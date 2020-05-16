# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'svm_classifier.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SVM_Classifier_Dialog(object):
    def setupUi(self, SVM_Classifier_Dialog):
        SVM_Classifier_Dialog.setObjectName("SVM_Classifier_Dialog")
        SVM_Classifier_Dialog.resize(490, 950)
        self.verticalLayout = QtWidgets.QVBoxLayout(SVM_Classifier_Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_1 = QtWidgets.QGroupBox(SVM_Classifier_Dialog)
        self.groupBox_1.setObjectName("groupBox_1")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_1)
        self.verticalLayout_3.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_3.setSpacing(2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_1)
        self.groupBox.setEnabled(True)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setVerticalSpacing(3)
        self.gridLayout.setObjectName("gridLayout")
        self.random_search_cb = QtWidgets.QCheckBox(self.groupBox)
        self.random_search_cb.setChecked(False)
        self.random_search_cb.setObjectName("random_search_cb")
        self.gridLayout.addWidget(self.random_search_cb, 0, 0, 1, 1)
        self.grid_search_cb = QtWidgets.QCheckBox(self.groupBox)
        self.grid_search_cb.setChecked(True)
        self.grid_search_cb.setObjectName("grid_search_cb")
        self.gridLayout.addWidget(self.grid_search_cb, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setToolTip("")
        self.label.setStatusTip("")
        self.label.setStyleSheet("font-size:12px;")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.cv_folds_sp = QtWidgets.QSpinBox(self.groupBox)
        self.cv_folds_sp.setMaximum(10)
        self.cv_folds_sp.setProperty("value", 5)
        self.cv_folds_sp.setObjectName("cv_folds_sp")
        self.gridLayout.addWidget(self.cv_folds_sp, 1, 1, 1, 1, QtCore.Qt.AlignLeft)
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_7.setMaximumSize(QtCore.QSize(16777215, 50))
        self.groupBox_7.setObjectName("groupBox_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.groupBox_7)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.linear_kernel_cb = QtWidgets.QCheckBox(self.groupBox_7)
        self.linear_kernel_cb.setObjectName("linear_kernel_cb")
        self.horizontalLayout_7.addWidget(self.linear_kernel_cb)
        self.poly_kernel_cb = QtWidgets.QCheckBox(self.groupBox_7)
        self.poly_kernel_cb.setObjectName("poly_kernel_cb")
        self.horizontalLayout_7.addWidget(self.poly_kernel_cb)
        self.rbf_kernel_cb = QtWidgets.QCheckBox(self.groupBox_7)
        self.rbf_kernel_cb.setChecked(True)
        self.rbf_kernel_cb.setObjectName("rbf_kernel_cb")
        self.horizontalLayout_7.addWidget(self.rbf_kernel_cb)
        self.sigmoid_kernel_cb = QtWidgets.QCheckBox(self.groupBox_7)
        self.sigmoid_kernel_cb.setObjectName("sigmoid_kernel_cb")
        self.horizontalLayout_7.addWidget(self.sigmoid_kernel_cb)
        self.gridLayout.addWidget(self.groupBox_7, 2, 0, 1, 2)
        self.c_group = QtWidgets.QGroupBox(self.groupBox)
        self.c_group.setMaximumSize(QtCore.QSize(16777215, 50))
        self.c_group.setObjectName("c_group")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.c_group)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_24 = QtWidgets.QLabel(self.c_group)
        self.label_24.setMinimumSize(QtCore.QSize(73, 0))
        self.label_24.setStyleSheet("font-size:12px;")
        self.label_24.setObjectName("label_24")
        self.horizontalLayout_3.addWidget(self.label_24)
        self.label_22 = QtWidgets.QLabel(self.c_group)
        self.label_22.setStyleSheet("font-size:12px;")
        self.label_22.setObjectName("label_22")
        self.horizontalLayout_3.addWidget(self.label_22, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.c_start_sp = QtWidgets.QSpinBox(self.c_group)
        self.c_start_sp.setMinimum(-6)
        self.c_start_sp.setMaximum(0)
        self.c_start_sp.setProperty("value", -2)
        self.c_start_sp.setObjectName("c_start_sp")
        self.horizontalLayout_3.addWidget(self.c_start_sp)
        self.label_23 = QtWidgets.QLabel(self.c_group)
        self.label_23.setStyleSheet("font-size:12px;")
        self.label_23.setObjectName("label_23")
        self.horizontalLayout_3.addWidget(self.label_23, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.c_end_sp = QtWidgets.QSpinBox(self.c_group)
        self.c_end_sp.setMinimum(1)
        self.c_end_sp.setMaximum(6)
        self.c_end_sp.setProperty("value", 2)
        self.c_end_sp.setObjectName("c_end_sp")
        self.horizontalLayout_3.addWidget(self.c_end_sp)
        self.label_21 = QtWidgets.QLabel(self.c_group)
        self.label_21.setStyleSheet("font-size:12px;")
        self.label_21.setObjectName("label_21")
        self.horizontalLayout_3.addWidget(self.label_21, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.c_num_sp = QtWidgets.QSpinBox(self.c_group)
        self.c_num_sp.setMinimum(5)
        self.c_num_sp.setMaximum(100)
        self.c_num_sp.setSingleStep(5)
        self.c_num_sp.setProperty("value", 5)
        self.c_num_sp.setObjectName("c_num_sp")
        self.horizontalLayout_3.addWidget(self.c_num_sp)
        self.gridLayout.addWidget(self.c_group, 3, 0, 1, 2)
        self.gamma_group = QtWidgets.QGroupBox(self.groupBox)
        self.gamma_group.setMaximumSize(QtCore.QSize(16777215, 50))
        self.gamma_group.setObjectName("gamma_group")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.gamma_group)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_20 = QtWidgets.QLabel(self.gamma_group)
        self.label_20.setStyleSheet("font-size:12px;")
        self.label_20.setObjectName("label_20")
        self.horizontalLayout_5.addWidget(self.label_20)
        self.label_17 = QtWidgets.QLabel(self.gamma_group)
        self.label_17.setStyleSheet("font-size:12px;")
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_5.addWidget(self.label_17, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.gamma_start_sp = QtWidgets.QSpinBox(self.gamma_group)
        self.gamma_start_sp.setMinimum(-5)
        self.gamma_start_sp.setMaximum(5)
        self.gamma_start_sp.setProperty("value", -2)
        self.gamma_start_sp.setObjectName("gamma_start_sp")
        self.horizontalLayout_5.addWidget(self.gamma_start_sp)
        self.label_18 = QtWidgets.QLabel(self.gamma_group)
        self.label_18.setStyleSheet("font-size:12px;")
        self.label_18.setObjectName("label_18")
        self.horizontalLayout_5.addWidget(self.label_18, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.gamma_end_sp = QtWidgets.QSpinBox(self.gamma_group)
        self.gamma_end_sp.setMinimum(1)
        self.gamma_end_sp.setMaximum(6)
        self.gamma_end_sp.setProperty("value", 2)
        self.gamma_end_sp.setObjectName("gamma_end_sp")
        self.horizontalLayout_5.addWidget(self.gamma_end_sp)
        self.label_19 = QtWidgets.QLabel(self.gamma_group)
        self.label_19.setStyleSheet("font-size:12px;")
        self.label_19.setObjectName("label_19")
        self.horizontalLayout_5.addWidget(self.label_19, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.gamma_num_sp = QtWidgets.QSpinBox(self.gamma_group)
        self.gamma_num_sp.setMinimum(5)
        self.gamma_num_sp.setMaximum(100)
        self.gamma_num_sp.setSingleStep(5)
        self.gamma_num_sp.setProperty("value", 5)
        self.gamma_num_sp.setObjectName("gamma_num_sp")
        self.horizontalLayout_5.addWidget(self.gamma_num_sp)
        self.gridLayout.addWidget(self.gamma_group, 4, 0, 1, 2)
        self.degree_group = QtWidgets.QGroupBox(self.groupBox)
        self.degree_group.setMaximumSize(QtCore.QSize(16777215, 50))
        self.degree_group.setObjectName("degree_group")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.degree_group)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_16 = QtWidgets.QLabel(self.degree_group)
        self.label_16.setStyleSheet("font-size:12px;")
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_4.addWidget(self.label_16)
        self.label_2 = QtWidgets.QLabel(self.degree_group)
        self.label_2.setMaximumSize(QtCore.QSize(48, 16777215))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.degree_start_sp = QtWidgets.QSpinBox(self.degree_group)
        self.degree_start_sp.setMinimum(1)
        self.degree_start_sp.setMaximum(1)
        self.degree_start_sp.setProperty("value", 1)
        self.degree_start_sp.setObjectName("degree_start_sp")
        self.horizontalLayout_4.addWidget(self.degree_start_sp)
        self.label_9 = QtWidgets.QLabel(self.degree_group)
        self.label_9.setStyleSheet("font-size:12px;")
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_4.addWidget(self.label_9, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.degree_end_sp = QtWidgets.QSpinBox(self.degree_group)
        self.degree_end_sp.setMinimum(2)
        self.degree_end_sp.setMaximum(5)
        self.degree_end_sp.setProperty("value", 2)
        self.degree_end_sp.setObjectName("degree_end_sp")
        self.horizontalLayout_4.addWidget(self.degree_end_sp)
        self.label_15 = QtWidgets.QLabel(self.degree_group)
        self.label_15.setStyleSheet("font-size:12px;")
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_4.addWidget(self.label_15, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.degree_num_sp = QtWidgets.QSpinBox(self.degree_group)
        self.degree_num_sp.setMinimum(1)
        self.degree_num_sp.setMaximum(4)
        self.degree_num_sp.setSingleStep(5)
        self.degree_num_sp.setProperty("value", 2)
        self.degree_num_sp.setObjectName("degree_num_sp")
        self.horizontalLayout_4.addWidget(self.degree_num_sp)
        self.gridLayout.addWidget(self.degree_group, 5, 0, 1, 2)
        self.coef_group = QtWidgets.QGroupBox(self.groupBox)
        self.coef_group.setMaximumSize(QtCore.QSize(16777215, 50))
        self.coef_group.setObjectName("coef_group")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.coef_group)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_13 = QtWidgets.QLabel(self.coef_group)
        self.label_13.setStyleSheet("font-size:12px;")
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_6.addWidget(self.label_13)
        self.label_12 = QtWidgets.QLabel(self.coef_group)
        self.label_12.setStyleSheet("font-size:12px;")
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_6.addWidget(self.label_12, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.coef_start_sp = QtWidgets.QSpinBox(self.coef_group)
        self.coef_start_sp.setMinimum(-6)
        self.coef_start_sp.setMaximum(-1)
        self.coef_start_sp.setProperty("value", -2)
        self.coef_start_sp.setObjectName("coef_start_sp")
        self.horizontalLayout_6.addWidget(self.coef_start_sp)
        self.label_8 = QtWidgets.QLabel(self.coef_group)
        self.label_8.setStyleSheet("font-size:12px;")
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_6.addWidget(self.label_8, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.coef_end_sp = QtWidgets.QSpinBox(self.coef_group)
        self.coef_end_sp.setMinimum(1)
        self.coef_end_sp.setMaximum(6)
        self.coef_end_sp.setProperty("value", 2)
        self.coef_end_sp.setObjectName("coef_end_sp")
        self.horizontalLayout_6.addWidget(self.coef_end_sp)
        self.label_11 = QtWidgets.QLabel(self.coef_group)
        self.label_11.setStyleSheet("font-size:12px;")
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_6.addWidget(self.label_11, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.coef_num_sp = QtWidgets.QSpinBox(self.coef_group)
        self.coef_num_sp.setMinimum(0)
        self.coef_num_sp.setMaximum(100)
        self.coef_num_sp.setSingleStep(5)
        self.coef_num_sp.setProperty("value", 0)
        self.coef_num_sp.setObjectName("coef_num_sp")
        self.horizontalLayout_6.addWidget(self.coef_num_sp)
        self.gridLayout.addWidget(self.coef_group, 6, 0, 1, 2)
        self.balanced_class_weight_cb = QtWidgets.QCheckBox(self.groupBox)
        self.balanced_class_weight_cb.setObjectName("balanced_class_weight_cb")
        self.gridLayout.addWidget(self.balanced_class_weight_cb, 7, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.groupBox)
        self.label_10.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 8, 0, 1, 1)
        self.scoring_comb = QtWidgets.QComboBox(self.groupBox)
        self.scoring_comb.setObjectName("scoring_comb")
        self.scoring_comb.addItem("")
        self.scoring_comb.setItemText(0, "")
        self.gridLayout.addWidget(self.scoring_comb, 8, 1, 1, 1)
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
        self.horizontalLayout_2.addWidget(self.display_model_name_label)
        self.open_model_btn = Open_Model_Btn(self.groupBox_4)
        self.open_model_btn.setEnabled(False)
        self.open_model_btn.setObjectName("open_model_btn")
        self.horizontalLayout_2.addWidget(self.open_model_btn)
        self.verticalLayout_3.addWidget(self.groupBox_4)
        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout.addWidget(self.groupBox_1)
        self.groupBox_2 = QtWidgets.QGroupBox(SVM_Classifier_Dialog)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.save_model_btn = Save_Model_Btn(self.groupBox_2)
        self.save_model_btn.setEnabled(False)
        self.save_model_btn.setObjectName("save_model_btn")
        self.gridLayout_2.addWidget(self.save_model_btn, 5, 0, 1, 1)
        self.plot_roc_cb = QtWidgets.QCheckBox(self.groupBox_2)
        self.plot_roc_cb.setChecked(True)
        self.plot_roc_cb.setObjectName("plot_roc_cb")
        self.gridLayout_2.addWidget(self.plot_roc_cb, 0, 0, 1, 1)
        self.output_cla_rep_cb = QtWidgets.QCheckBox(self.groupBox_2)
        self.output_cla_rep_cb.setChecked(True)
        self.output_cla_rep_cb.setObjectName("output_cla_rep_cb")
        self.gridLayout_2.addWidget(self.output_cla_rep_cb, 1, 0, 1, 1)
        self.save_model_label = QtWidgets.QLabel(self.groupBox_2)
        self.save_model_label.setObjectName("save_model_label")
        self.gridLayout_2.addWidget(self.save_model_label, 5, 1, 1, 2, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.save_model_cb = QtWidgets.QCheckBox(self.groupBox_2)
        self.save_model_cb.setObjectName("save_model_cb")
        self.gridLayout_2.addWidget(self.save_model_cb, 4, 0, 1, 1)
        self.output_confusion_cb = QtWidgets.QCheckBox(self.groupBox_2)
        self.output_confusion_cb.setChecked(True)
        self.output_confusion_cb.setObjectName("output_confusion_cb")
        self.gridLayout_2.addWidget(self.output_confusion_cb, 1, 1, 1, 2)
        self.save_file_btn = Save_File_Btn(self.groupBox_2)
        self.save_file_btn.setEnabled(False)
        self.save_file_btn.setObjectName("save_file_btn")
        self.gridLayout_2.addWidget(self.save_file_btn, 3, 0, 1, 1)
        self.save_file_cb = QtWidgets.QCheckBox(self.groupBox_2)
        self.save_file_cb.setObjectName("save_file_cb")
        self.gridLayout_2.addWidget(self.save_file_cb, 2, 0, 1, 2)
        self.save_file_label = QtWidgets.QLabel(self.groupBox_2)
        self.save_file_label.setObjectName("save_file_label")
        self.gridLayout_2.addWidget(self.save_file_label, 3, 1, 1, 2, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(SVM_Classifier_Dialog)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox_3)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_2.addWidget(self.textBrowser)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.widget = QtWidgets.QWidget(SVM_Classifier_Dialog)
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

        self.retranslateUi(SVM_Classifier_Dialog)
        QtCore.QMetaObject.connectSlotsByName(SVM_Classifier_Dialog)

    def retranslateUi(self, SVM_Classifier_Dialog):
        _translate = QtCore.QCoreApplication.translate
        SVM_Classifier_Dialog.setWindowTitle(_translate("SVM_Classifier_Dialog", "SVM Classifier"))
        self.groupBox_1.setTitle(_translate("SVM_Classifier_Dialog", "parameter setting"))
        self.groupBox.setTitle(_translate("SVM_Classifier_Dialog", "train a model"))
        self.random_search_cb.setText(_translate("SVM_Classifier_Dialog", "RandomizedSearchCV"))
        self.grid_search_cb.setText(_translate("SVM_Classifier_Dialog", "GridSearchCV"))
        self.label.setText(_translate("SVM_Classifier_Dialog", "Folds for Cross Validatin"))
        self.groupBox_7.setTitle(_translate("SVM_Classifier_Dialog", "Kernels"))
        self.linear_kernel_cb.setText(_translate("SVM_Classifier_Dialog", "linear"))
        self.poly_kernel_cb.setText(_translate("SVM_Classifier_Dialog", "polynomial"))
        self.rbf_kernel_cb.setText(_translate("SVM_Classifier_Dialog", "rbf"))
        self.sigmoid_kernel_cb.setText(_translate("SVM_Classifier_Dialog", "sigmoid "))
        self.c_group.setTitle(_translate("SVM_Classifier_Dialog", "C parameter for all kernal"))
        self.label_24.setText(_translate("SVM_Classifier_Dialog", "Cs from "))
        self.label_22.setText(_translate("SVM_Classifier_Dialog", "10 ^"))
        self.label_23.setText(_translate("SVM_Classifier_Dialog", "to 10 ^"))
        self.label_21.setText(_translate("SVM_Classifier_Dialog", "select"))
        self.gamma_group.setTitle(_translate("SVM_Classifier_Dialog", "γ for all kernel excluding linear"))
        self.label_20.setText(_translate("SVM_Classifier_Dialog", "Gammas from "))
        self.label_17.setText(_translate("SVM_Classifier_Dialog", "10 ^"))
        self.label_18.setText(_translate("SVM_Classifier_Dialog", "to 10 ^"))
        self.label_19.setText(_translate("SVM_Classifier_Dialog", "select"))
        self.degree_group.setTitle(_translate("SVM_Classifier_Dialog", "Degree for poly kernel"))
        self.label_16.setText(_translate("SVM_Classifier_Dialog", "Degrees from "))
        self.label_9.setText(_translate("SVM_Classifier_Dialog", "to"))
        self.label_15.setText(_translate("SVM_Classifier_Dialog", "select"))
        self.coef_group.setTitle(_translate("SVM_Classifier_Dialog", "Independent coefficient for poly and sidmoid kernel"))
        self.label_13.setText(_translate("SVM_Classifier_Dialog", "coef0s from "))
        self.label_12.setText(_translate("SVM_Classifier_Dialog", "10 ^"))
        self.label_8.setText(_translate("SVM_Classifier_Dialog", "to 10 ^"))
        self.label_11.setText(_translate("SVM_Classifier_Dialog", "select"))
        self.balanced_class_weight_cb.setText(_translate("SVM_Classifier_Dialog", "balanced class weight"))
        self.label_10.setText(_translate("SVM_Classifier_Dialog", "CV target scoring metrix"))
        self.groupBox_4.setTitle(_translate("SVM_Classifier_Dialog", "Or load a model"))
        self.load_model_cb.setText(_translate("SVM_Classifier_Dialog", "Load Model"))
        self.display_model_name_label.setText(_translate("SVM_Classifier_Dialog", "No File Selcted"))
        self.open_model_btn.setText(_translate("SVM_Classifier_Dialog", "Load model(joblib file)"))
        self.groupBox_2.setTitle(_translate("SVM_Classifier_Dialog", "Out Parameters"))
        self.save_model_btn.setText(_translate("SVM_Classifier_Dialog", "Save the Model"))
        self.plot_roc_cb.setText(_translate("SVM_Classifier_Dialog", "Plot ROC on training and testing datsets"))
        self.output_cla_rep_cb.setText(_translate("SVM_Classifier_Dialog", "Output classification report"))
        self.save_model_label.setText(_translate("SVM_Classifier_Dialog", "No Directory selected"))
        self.save_model_cb.setText(_translate("SVM_Classifier_Dialog", "Save fitted model into"))
        self.output_confusion_cb.setText(_translate("SVM_Classifier_Dialog", "Output confusion matrix"))
        self.save_file_btn.setText(_translate("SVM_Classifier_Dialog", "Save the Files"))
        self.save_file_cb.setText(_translate("SVM_Classifier_Dialog", "Save (transformed) training (and testng) file(s) to"))
        self.save_file_label.setText(_translate("SVM_Classifier_Dialog", "No Directory selected"))
        self.groupBox_3.setTitle(_translate("SVM_Classifier_Dialog", "Output field"))
        self.apply_btn.setText(_translate("SVM_Classifier_Dialog", "Apply"))
        self.reset_btn.setText(_translate("SVM_Classifier_Dialog", "Reset From Previous"))
        self.finish_btn.setText(_translate("SVM_Classifier_Dialog", "Finish"))
from open_model_btn import Open_Model_Btn
from save_file_btn import Save_File_Btn
from save_model_btn import Save_Model_Btn


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SVM_Classifier_Dialog = QtWidgets.QDialog()
    ui = Ui_SVM_Classifier_Dialog()
    ui.setupUi(SVM_Classifier_Dialog)
    SVM_Classifier_Dialog.show()
    sys.exit(app.exec_())
