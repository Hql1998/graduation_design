from GUI_part.random_forest_classifier_ui import Ui_Dialog
from PyQt5 import QtWidgets
from PyQt5.Qt import QDialog, QErrorMessage, qErrnoWarning, QButtonGroup, QDoubleValidator, QValidator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from joblib import dump, load
from print_to_log import *
from public_functions import plot_confusion_matrix_public, print_classification_report_public, save_file_public, check_data_public, check_data_model_compatible_public


class Validator01(QDoubleValidator):
    def validate(self, input_text, pos):
        status, input_text, pos = super().validate(input_text, pos)
        if QValidator.Acceptable == status:
            if 0<float(input_text)<1:
                return (status, input_text, pos)
            elif 0==float(input_text):
                return (QValidator.Intermediate, input_text, pos)
            else:
                return (QValidator.Invalid, input_text, pos-1)
        else:
            return (status, input_text, pos)

    def fixup(self, input_text):
        return "0.0001"


class Random_Forest_Classifier_Dialog(QDialog, Ui_Dialog):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setupUi(self)
        self.adjust_sbutile()
        self.data = {}
        self.cv_model = None
        self.model = None
        self.run_index = 0
        self.have_test = True
        self.multiclass = False
        self.save_model_later = False
        self.save_file_later = False
        self.class_name = []

        self.parentWidget().state_changed_handler("start")

    def adjust_sbutile(self):
        self.setStyleSheet("background:None;")

        btn_group = QButtonGroup(self)
        btn_group.addButton(self.random_search_cb)
        btn_group.addButton(self.grid_search_cb)
        validator01 = Validator01()
        self.feature_importance_le.setValidator(validator01)

    def get_y_labels(self):
        if self.data["train_y"] is not None:
            y_series = pd.Series(self.data["train_y"].to_numpy().reshape(len(self.data["train_y"].to_numpy()),))
            unique_y = y_series.value_counts().shape[0]
            if unique_y <= 2:
                self.multiclass = False
                self.scoring_comb.clear()
                self.scoring_comb.addItems(["accuracy", "roc_auc", "precision", "recall"])
            else:
                self.multiclass = True
                self.scoring_comb.clear()
                self.scoring_comb.addItems(["accuracy", "roc_auc_ovr", "precision_micro", "recall_micro"])

    def load_model_cb_toggled_handler(self):
        print("load_model clicked")
        if self.load_model_cb.checkState():
            self.open_model_btn.setEnabled(True)
            self.groupBox.setEnabled(False)
            # 按钮被点击会阻塞消息，操作，所以需要放到最后
            self.open_model_btn.click()

        else:
            print("ok")
            self.open_model_btn.setEnabled(False)
            self.groupBox.setEnabled(True)

    def open_model_btn_clicked_handler(self):

        if self.open_model_btn.open_result[0] != "":
            self.display_model_name_label.setText("/".join(self.open_model_btn.open_result[0].split("/")[0:-1]))
            self.display_model_name_label.adjustSize()
            self.load_a_model()
        else:
            self.display_model_name_label.setText("No Directory selected")
            self.display_model_name_label.adjustSize()
            self.load_model_cb.setChecked(False)

    def load_a_model(self):
        if self.display_model_name_label.text() != "No Directory selected" and self.load_model_cb.checkState():
            file_path = self.open_model_btn.open_result[0]
            self.cv_model = load(file_path)
        else:
            QErrorMessage.qtHandler()
            qErrnoWarning("you didn't select the checkbox or a valid file")

    def save_model_btn_clicked_handler(self):
        if self.save_model_btn.open_result[0] != "":
            self.save_model_label.setText("/".join(self.save_model_btn.open_result[0].split("/")[0:-1]))
            self.save_model_label.adjustSize()
            self.save_a_model()
        else:
            self.save_model_label.setText("No Directory selected")
            self.save_model_label.adjustSize()
            self.save_model_cb.setChecked(False)

    def save_model_cb_toggled_handler(self):
        if self.save_model_cb.checkState():
            self.save_model_btn.setEnabled(True)
            print("clicked")
            self.save_model_btn.click()
            print("clicked")

        else:
            self.save_model_btn.setEnabled(False)

    def save_a_model(self):
        model = self.cv_model
        if self.save_model_label.text() != "No Directory selected" and self.save_model_cb.checkState() and model is not None:
            file_path = self.save_model_btn.open_result[0]
            dump(model, file_path)
            self.save_model_later = False
        else:
            self.save_model_later = True
            # QErrorMessage.qtHandler()
            # qErrnoWarning("you don't have a trained model or ou didn't select the checkbox")

    def save_file_btn_clicked_handler(self):
        if self.save_file_btn.open_result[0] != "":
            self.save_file_label.setText("/".join(self.save_file_btn.open_result[0].split("/")[0:-1]))
            self.save_file_label.adjustSize()
            self.save_file()
        else:
            self.save_file_label.setText("No Directory selected")
            self.save_file_label.adjustSize()
            self.save_file_cb.setChecked(False)

    def save_file_cb_toggled_handler(self):
        if self.save_file_cb.checkState():
            self.save_file_btn.setEnabled(True)
            self.save_file_btn.click()
        else:
            self.save_file_btn.setEnabled(False)

    def save_file(self):

        if self.save_file_label.text() != "No Directory selected" and self.save_file_cb.checkState():
            save_file_public(open_result=self.save_file_btn.open_result,
                             data=self.data,
                             have_test=self.have_test,
                             widget_name=self.parentWidget().class_name,
                             run_index=self.run_index)
            self.save_file_later = False
        else:
            self.save_file_later = True
            # QErrorMessage.qtHandler()
            # qErrnoWarning("you don't have a trained model or ou didn't select the checkbox")

    def feature_filter_toggled_handler(self):
        if self.model is not None and self.model.n_features_ == self.data["train_x"].shape[1]:
            self.transform_data()

    def train_a_model(self):
        if self.data["train_y"] is not None:
            print("you got a response variable")
            train_x = self.data["train_x"]
            train_y = self.data["train_y"]

            tree_num_start = self.tree_num_start_sp.value()
            tree_num_end = self.tree_end_end_sp.value()
            tree_num = self.tree_num_sp.value()

            feature_length = train_x.shape[1]
            feature_prop_start = self.feature_prop_start_dsp.value() * feature_length
            feature_prop_end = self.feature_prop_end_dsp.value() * feature_length
            if feature_prop_end <= feature_prop_start:
                return True
            feature_prop_num = self.feature_prop_num_sp.value()

            max_depth_start = self.max_depth_start_sp.value()
            max_depth_end = self.max_depth_end_sp.value()
            max_depth_num = self.max_depth_num_sp.value()

            min_split_start = self.min_split_start_sp.value()
            min_split_end = self.min_split_end_sp.value()
            min_split_num = self.min_split_num_sp.value()

            min_leaf_start = self.min_leaf_start_sp.value()
            min_leaf_end = self.min_leaf_end_sp.value()
            min_leaf_num = self.min_leaf_num_sp.value()

            scoring = self.scoring_comb.currentText()
            cv_folds = self.cv_folds_sp.value()

            n_estimators = [int(x) for x in np.linspace(start=tree_num_start, stop=tree_num_end, num=tree_num)]

            max_features = np.array([int(x) for x in np.linspace(start=feature_prop_start, stop=feature_prop_end, num=feature_prop_num)])
            max_features = max_features[max_features > 0]
            # max_features = np.append(max_features, "auto")


            max_depth = [int(x) for x in np.linspace(start=max_depth_start, stop=max_depth_end, num=max_depth_num)]
            max_depth.append(None)
            min_samples_split = np.array([int(x) for x in np.linspace(start=min_split_start, stop=min_split_end, num=min_split_num)])
            min_samples_split = min_samples_split[min_samples_split < 0.5*train_x.shape[0]]
            min_samples_leaf = np.array([int(x) for x in np.linspace(start=min_leaf_start, stop=min_leaf_end, num=min_leaf_num)])
            min_samples_leaf = min_samples_leaf[min_samples_leaf < 0.5 * train_x.shape[0]]

            grid_parameters = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf}
            print(grid_parameters)

            if self.balanced_class_weight_cb.checkState():
                rf = RandomForestClassifier(class_weight="balanced")
            else:
                rf = RandomForestClassifier()

            print("computing")

            if self.random_search_cb.checkState():
                random_cv = RandomizedSearchCV(estimator=rf, param_distributions=grid_parameters, cv=cv_folds, scoring=scoring, random_state=42) # n_iter=100,
                random_cv.fit(train_x.to_numpy(), train_y.to_numpy().reshape(train_y.to_numpy().shape[0], ))
                self.cv_model = random_cv
            elif self.grid_search_cb.checkState():
                grid_cv = GridSearchCV(estimator=rf, param_grid=grid_parameters, cv=cv_folds, scoring=scoring)
                grid_cv.fit(train_x.to_numpy(), train_y.to_numpy().reshape(train_y.to_numpy().shape[0], ))
                self.cv_model = grid_cv
            print("training model finished")

            return False

    def plot_ROC_just_curve(self, model_name, X, y):
        classifier = self.model
        color = {'Random Forest - test': 'darkgreen', 'Random Forest - train': 'darkblue'}
        probas = classifier.predict_proba(X)
        print("y_true", y.to_numpy().reshape(y.shape[0],).shape, "y_prob", probas.shape)
        index = 1
        for i in range(self.model.n_classes_):
            if self.model.classes_[i] == 1:
                index = i
        fpr, tpr, thresholds = roc_curve(y.to_numpy().reshape(y.shape[0],), probas[:,index])
        roc_auc = auc(fpr, tpr)
        print_to_tb(self.textBrowser,model_name + r' ROC (AUC: %0.2f)' % (roc_auc))
        plt.plot(fpr, tpr, color=color[model_name], label=model_name + r' ROC (AUC: %0.2f)' % (roc_auc), lw=2, alpha=.9)
        return None

    def plot_ROC(self):

        plt.figure(num = "Random Forest Classifier ROC"+" the "+str(self.run_index)+" run", figsize=(5, 5))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Chance', alpha=.8)
        self.plot_ROC_just_curve("Random Forest - train", self.data["train_x"], self.data["train_y"])
        if self.have_test:
            self.plot_ROC_just_curve("Random Forest - test", self.data["test_x"], self.data["test_y"])

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('1 - Specificity', fontsize=7)
        plt.ylabel('Sensitivity', fontsize=7)
        # plt.title(title +' ROC curve using test set')
        plt.legend(loc="lower right", prop={'size': 7})
        plt.show(block=False)

    def plot_ROC_multiclass(self, testing=False):
        classifier = self.model
        n_classes = len(classifier.classes_)
        if testing:
            y_score = classifier.predict_proba(self.data["test_x"])
            y = label_binarize(self.data["test_y"], classes=classifier.classes_)
        else:
            y_score = classifier.predict_proba(self.data["train_x"])
            y = label_binarize(self.data["train_y"], classes=classifier.classes_)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        print(n_classes)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        if testing:
            plt.figure(num="ROC on testing dataset"+" the "+str(self.run_index)+" run",figsize=(6,6))
        else:
            plt.figure(num="ROC on training dataset"+" the "+str(self.run_index)+" run",figsize=(6,6))
        linewidth = 2

        if testing:
            prefix_tb = "testing file"
        else:
            prefix_tb = "traing file"
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=linewidth)
        print_to_tb(self.textBrowser, prefix_tb, 'micro-average AUC is {0:0.2f})'.format(roc_auc["micro"]))
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=linewidth)
        print_to_tb(self.textBrowser, prefix_tb, 'macro-average AUC is {0:0.2f})'.format(roc_auc["macro"]))
        colors = cycle(
            [(0.878, 0.4627, 0.3529), (0.392, 0, 0), (0.2, 0.21, 0.38), (0.4, 0.843, 0.513), (0.274, 0.51, 0.878),
             (0.21, 0.6627, 0.576),(0, 0.3568, 0.61), (0.42, 0.8588, 0.7682), (0.8, 0.43, 0.666),(0.59, 0.2549, 0.1)])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=linewidth,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))
            print_to_tb(self.textBrowser, prefix_tb, 'AUC of class {0} is {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=linewidth)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show(block=False)#block=False

    def print_result(self):

        cv_model = self.cv_model
        self.model = cv_model.best_estimator_
        best_estimator = self.model

        self.class_name = ["class " + str(i) for i in best_estimator.classes_]
        print(self.class_name)
        if best_estimator.n_features_ != self.data["train_x"].shape[1]:
            QErrorMessage.qtHandler()
            qErrnoWarning("the data shape dosen't match the model")
            return True

        print_tb_header(self.textBrowser, self.run_index)
        for key in cv_model.best_params_.keys():
            print_to_tb(self.textBrowser,key, cv_model.best_params_[key])

        for index in range(best_estimator.n_features_):
            print_to_tb(self.textBrowser,"the "+str(index+1)+" feature: ",self.data["train_x"].columns[index], best_estimator.feature_importances_[index])

        print_to_tb(self.textBrowser, "best score during CV", cv_model.best_score_)
        if self.multiclass and self.plot_roc_cb.checkState():
            self.plot_ROC_multiclass(False)
            if self.have_test:
                self.plot_ROC_multiclass(True)
        elif not self.multiclass and self.plot_roc_cb.checkState():
            self.plot_ROC()

    def print_classification_report(self):
        print_classification_report_public(model=self.model, data=self.data, class_name=self.class_name, textBrowser=self.textBrowser, have_test=self.have_test)

    def plot_confusion_matrix(self):
        plot_confusion_matrix_public(model=self.model, model_name="Random_Forest", data=self.data, class_name=self.class_name, run_index=self.run_index, have_test=self.have_test)

    def transform_data(self):
        threshold = float(self.feature_importance_le.text())
        sfm = SelectFromModel(estimator=self.model, prefit=True, threshold=threshold)
        self.data["train_x"] = pd.DataFrame(sfm.transform(self.data["train_x"]), index=self.data["train_x"].index, columns=self.data["train_x"].columns[sfm.get_support()])
        print_to_tb(self.textBrowser, "number of finially selected features：", len(sfm.get_support()[sfm.get_support() == True]), "/", self.model.n_features_)
        print_log_header(self.parentWidget().class_name)
        print_to_log("train_x" + str(self.data["train_x"].shape))
        print_to_log("train_y" + str(self.data["train_y"].shape))
        if self.have_test:
            self.data["test_x"] = pd.DataFrame(sfm.transform(self.data["test_x"]), index=self.data["test_x"].index,
                                               columns=self.data["test_x"].columns[sfm.get_support()])
            print_to_log("test_x" + str(self.data["test_x"].shape))
            print_to_log("test_y" + str(self.data["test_y"].shape))

    def apply_handler(self):

        self.parentWidget().state_changed_handler("processing")

        no_y_flag, self.have_test = check_data_public(self.data)
        if no_y_flag:
            self.parentWidget().state_changed_handler("unprepared")
            return

        self.run_index += 1
        if self.load_model_cb.checkState():
            pass
        else:
            if self.train_a_model():
                return None

        if check_data_model_compatible_public(self.data, self.cv_model,"n_features_", cv=True):
            return None

        self.print_result()

        if self.output_cla_rep_cb.checkState():
            self.print_classification_report()

        if self.output_confusion_cb.checkState():
            self.plot_confusion_matrix()

        if self.save_model_cb.checkState() and self.save_model_later:
            self.save_a_model()

        if self.feature_filter_cb.checkState():
            self.transform_data()

        if self.save_file_cb.checkState() and self.save_file_later:
            self.save_file()

        self.parentWidget().state_changed_handler("finish")
        plt.show()

    def finish_handler(self):

        if self.parent().next_widgets != []:
            for next_widget in self.parent().next_widgets:
                if next_widget.data is None:
                    next_widget.update_data_from_previous()
        self.hide()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = Random_Forest_Classifier_Dialog()
    Dialog.show()
    sys.exit(app.exec_())

