from GUI_part.naive_bayes_classifier_ui import Ui_Naive_Bayes_Dialog
from PyQt5 import QtWidgets
from PyQt5.Qt import QDialog, QErrorMessage, qErrnoWarning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from joblib import dump, load
from print_to_log import *
from public_functions import plot_confusion_matrix_public, print_classification_report_public, save_file_public, check_data_public, check_data_model_compatible_public,read_data_multiclass_public,read_data_public


class mixed_Naive_Bayes(BaseEstimator):
    def __init__(self, multinomial_var_list=[], bernoulli_var_list=[], gaussian_var_list=[]):
        self._estimator_type = "classifier"
        self.multinomial_var_list = multinomial_var_list
        self.bernoulli_var_list = bernoulli_var_list
        self.gaussian_var_list = gaussian_var_list
        self.fitted = False
        self.n_features_ = 0
        self.gnb = None
        self.bnb = None
        self.mnb = None
        self.classes_ = None

    def fit(self, X, y):

        if self.gaussian_var_list != []:
            gaussian_x = X[self.gaussian_var_list]
            self.gnb = GaussianNB()
            self.gnb.fit(gaussian_x, y)
            self.classes_ = self.gnb.classes_

        if self.bernoulli_var_list != []:
            bernoulli_x = X[self.bernoulli_var_list]
            self.bnb = BernoulliNB()
            self.bnb.fit(bernoulli_x, y)
            self.classes_ = self.bnb.classes_

        if self.multinomial_var_list != []:
            multinomial_x = X[self.multinomial_var_list]
            self.mnb = MultinomialNB()
            self.mnb.fit(multinomial_x, y)
            self.classes_ = self.mnb.classes_

        self.n_features_ = X.shape[1]
        self.fitted = True
        return self

    def predict(self, X):

        prob_mean = self.predict_proba(X)
        # [np.argmax(i) for i in prob_mean[:, :]]
        y = [self.classes_[np.argmax(i)] for i in prob_mean[:, :]]

        return y

    def predict_proba(self, X):
        if self.fitted:
            prob_list = []
            if self.gaussian_var_list != []:
                gaussian_x = X[self.gaussian_var_list]
                gnb_proba = self.gnb.predict_proba(gaussian_x)
                prob_list.append(gnb_proba)

            if self.bernoulli_var_list != []:
                bernoulli_x = X[self.bernoulli_var_list]
                bnb_proba = self.bnb.predict_proba(bernoulli_x)
                prob_list.append(bnb_proba)

            if self.multinomial_var_list != []:
                multinomial_x = X[self.multinomial_var_list]
                mnb_proba = self.mnb.predict_proba(multinomial_x)
                prob_list.append(mnb_proba)

            prob_mean = np.array(prob_list[0])
            for i in range(prob_mean.shape[1]):
                prob_mean[:, i] = np.array([j[:, i] for j in prob_list]).mean(axis=0)

            return prob_mean

class My_Gaussian_Naive_Bayes(GaussianNB):
    def __init__(self, multinomial_var_list=[], bernoulli_var_list=[], gaussian_var_list=[]):
        super().__init__()
        self.n_features_ = 0

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        return super().fit(X, y)

class Naive_Bayes_Classifier_Dialog(QDialog, Ui_Naive_Bayes_Dialog):

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

        # read_data_public(self.data)
        # read_data_multiclass_public(self.data)

        # self.get_y_labels()
        # self.apply_handler()

        self.parentWidget().state_changed_handler("start")

    def adjust_sbutile(self):

        self.setStyleSheet("background:None;")

        self.load_model_cb.toggled['bool'].connect(self.load_model_cb_toggled_handler)
        self.open_model_btn.clicked.connect(self.open_model_btn_clicked_handler)
        self.save_file_cb.toggled['bool'].connect(self.save_file_cb_toggled_handler)
        self.save_file_btn.clicked.connect(self.save_file_btn_clicked_handler)
        self.save_model_cb.toggled['bool'].connect(self.save_model_cb_toggled_handler)
        self.save_model_btn.clicked.connect(self.save_model_btn_clicked_handler)
        self.apply_btn.clicked.connect(self.apply_handler)
        self.finish_btn.clicked.connect(self.finish_handler)

        # self.multinomial_distribute_rb.setChecked(True)

    def get_y_labels(self):
        if self.data["train_y"] is not None:
            y_series = pd.Series(self.data["train_y"].to_numpy().reshape(len(self.data["train_y"].to_numpy()),))
            unique_y = y_series.value_counts().shape[0]
            if unique_y <= 2:
                self.multiclass = False
            else:
                self.multiclass = True

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
            self.model = load(file_path)
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
        model = self.model
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

    def transform_accrod_to_type(self):
        train_x = self.data["train_x"]
        train_y = self.data["train_y"]

        self.multinomial_var_list = []
        self.bernoulli_var_list = []
        self.gaussian_var_list = []

        for col in train_x.columns:
            if len(train_x[col].unique().tolist()) == 2:
                self.bernoulli_var_list.append(col)
            elif len(train_x[col].unique().tolist()) > 2:
                data_type = train_x[col].dtype

                if data_type == np.int_ or data_type == np.int64:
                    self.multinomial_var_list.append(col)
                else:
                    self.gaussian_var_list.append(col)

        print(self.multinomial_var_list)

        if self.gaussian_distribute_rb.isChecked():
            gaussian_x = train_x[self.gaussian_var_list]
            if gaussian_x.shape[1] != train_x.shape[1]:
                QErrorMessage.qtHandler()
                qErrnoWarning(
                    "some data may  be droped due to data type dosen't fit,keeped feature has length of {0},you may choose auto mode".format(
                        len(self.gaussian_var_list)))
            self.data["train_x"] = gaussian_x
            if self.have_test:
                self.data["test_x"] = self.data["test_x"][self.gaussian_var_list]

        elif self.multinomial_distribute_rb.isChecked():
            multinomial_x = train_x[self.multinomial_var_list]
            if multinomial_x.shape[1] != train_x.shape[1]:
                QErrorMessage.qtHandler()
                qErrnoWarning(
                    "some data may  be droped due to data type dosen't fit,keeped feature has length of {0},you may choose auto mode".format(
                        len(self.multinomial_var_list)))
            self.data["train_x"] = multinomial_x
            if self.have_test:
                self.data["test_x"] = self.data["test_x"][self.multinomial_var_list]

        elif self.bernoulli_distribute_rb.isChecked():
            bernoulli_x = train_x[self.bernoulli_var_list]
            if bernoulli_x.shape[1] != train_x.shape[1]:
                QErrorMessage.qtHandler()
                qErrnoWarning(
                    "some data may  be droped due to data type dosen't fit,keeped feature has length of {0},you may choose auto mode".format(
                        len(self.bernoulli_var_list)))
            self.data["train_x"] = bernoulli_x
            if self.have_test:
                self.data["test_x"] = self.data["test_x"][self.bernoulli_var_list]

    def train_a_model(self):

        train_x = self.data["train_x"]
        train_y = self.data["train_y"]

        model = None
        print("computing")

        if self.auto_distribute_rb.isChecked():
            mix_nb = mixed_Naive_Bayes(self.multinomial_var_list, self.bernoulli_var_list, self.gaussian_var_list)
            model = mix_nb.fit(train_x, train_y.to_numpy().reshape(train_y.to_numpy().shape[0], ))
        else:
            if self.gaussian_distribute_rb.isChecked() and len(self.gaussian_var_list) > 0:
                gnb = My_Gaussian_Naive_Bayes()
                model = gnb.fit(train_x.to_numpy(), train_y.to_numpy().reshape(train_y.to_numpy().shape[0], ))
            elif self.multinomial_distribute_rb.isChecked() and len(self.multinomial_var_list) > 0:
                mnb = MultinomialNB()
                model = mnb.fit(train_x.to_numpy(), train_y.to_numpy().reshape(train_y.to_numpy().shape[0], ))

            elif self.bernoulli_distribute_rb.isChecked() and len(self.bernoulli_var_list) > 0:
                bnb = BernoulliNB()
                model = bnb.fit(train_x.to_numpy(), train_y.to_numpy().reshape(train_y.to_numpy().shape[0], ))

        if model:
            self.model = model
            print_log_header("Naive Bayes Classifier")
            print_to_log("train x's shape is {0}".format(self.data["train_x"].shape))
            print_to_log("train y's shape is {0}".format(self.data["train_y"].shape))
            if self.have_test:
                print_to_log("test x's shape is {0}".format(self.data["test_x"].shape))
                print_to_log("test y's shape is {0}".format(self.data["test_y"].shape))
        else:
            QErrorMessage.qtHandler()
            qErrnoWarning("you didn't select a model when training")
            return True

        print("training model finished")
        return False

    def plot_ROC_just_curve(self, model_name, X, y):
        classifier = self.model
        color = {'Naive Bayes - test': 'darkgreen', 'Naive Bayes - train': 'darkblue'}
        probas = classifier.predict_proba(X)
        print("y_true", y.to_numpy().reshape(y.shape[0],).shape, "y_prob", probas.shape)

        fpr, tpr, thresholds = roc_curve(y.to_numpy().reshape(y.shape[0],), probas[:,1])
        roc_auc = auc(fpr, tpr)
        print_to_tb(self.textBrowser,model_name + r' ROC (AUC: %0.2f)' % (roc_auc))
        plt.plot(fpr, tpr, color=color[model_name], label=model_name + r' ROC (AUC: %0.2f)' % (roc_auc), lw=2, alpha=.9)
        return None

    def plot_ROC(self):

        plt.figure(num = "Naive Bayes Classifier ROC"+" the "+str(self.run_index)+" run", figsize=(5, 5))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Chance', alpha=.8)
        self.plot_ROC_just_curve("Naive Bayes - train", self.data["train_x"], self.data["train_y"])
        if self.have_test:
            self.plot_ROC_just_curve("Naive Bayes - test", self.data["test_x"], self.data["test_y"])

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
        colors = colors = cycle(
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

        best_estimator = self.model

        self.class_name = ["class " + str(i) for i in best_estimator.classes_]
        print(self.class_name)

        print_tb_header(self.textBrowser, self.run_index)

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

    def apply_handler(self):

        self.parentWidget().state_changed_handler("processing")

        no_y_flag, self.have_test = check_data_public(self.data)
        if no_y_flag:
            self.parentWidget().state_changed_handler("unprepared")
            return

        self.run_index += 1
        self.transform_accrod_to_type()

        if self.load_model_cb.checkState():
            pass
        else:
            if self.train_a_model():
                return None

        if check_data_model_compatible_public(self.data, self.model, "n_features_", cv=False):
            return None
        self.print_result()

        if self.output_cla_rep_cb.checkState():
            self.print_classification_report()

        if self.output_confusion_cb.checkState():
            self.plot_confusion_matrix()

        if self.save_model_cb.checkState() and self.save_model_later:
            self.save_a_model()

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
    self = Naive_Bayes_Classifier_Dialog()
    self.show()
    sys.exit(app.exec_())

