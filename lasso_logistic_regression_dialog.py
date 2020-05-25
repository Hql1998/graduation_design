from GUI_part.lasso_logistic_regression_ui import Ui_Dialog
from PyQt5 import QtWidgets
from PyQt5.Qt import QDialog,QErrorMessage, qErrnoWarning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from joblib import dump, load
from print_to_log import *
from public_functions import plot_confusion_matrix_public, print_classification_report_public, save_file_public, check_data_public,check_data_model_compatible_public


class Lasso_Logistic_Regression_Dialog(QDialog, Ui_Dialog):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setupUi(self)
        # self.data = self.parentWidget().data
        self.data = {}
        self.have_test = True
        self.LLRCV = None
        self.run_index= 0
        self.multiclass = False
        self.save_model_later = False
        self.save_file_later = False
        self.transformed = False
        self.class_name = []

        self.setStyleSheet("background:None;")

        self.parentWidget().state_changed_handler("start")

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
            self.LLRCV = load(file_path)
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
        model = self.LLRCV
        if self.save_model_label.text() != "No Directory selected" and self.save_model_cb.checkState() and model is not None:
            file_path = self.save_model_btn.open_result[0].replace(".joblib", "self.parentWidget().class_name"+"_the_"+ str(self.run_index) + "_run"+".joblib")
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

        if self.save_file_label.text() != "No Directory selected" and self.save_file_cb.checkState() and self.transformed:
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

    def train_a_model(self):
        if self.data["train_y"] is not None:
            print("you got a response variable")
            train_x = self.data["train_x"]
            train_y = self.data["train_y"]
            cv_folds = self.cv_folds_sp.value()
            grid_end_value = self.grid_end_sp.value()
            grid_number = self.grid_num_sp.value()
            grid_start_value = self.grid_start_sp.value()
            max_iter = 10**self.max_iter_sp.value()
            tolerance = 10**self.tol_sp.value()

            c_values = np.logspace(start=grid_start_value, stop=grid_end_value, num=grid_number, base=10)
            scoring = self.scoring_comb.currentText()
            print("okok")
            if self.balanced_class_weight_cb.checkState():
                LLRCV = LogisticRegressionCV(Cs=c_values, cv=cv_folds, penalty='l1', solver="saga", scoring=scoring, tol=tolerance,
                                             max_iter=max_iter, class_weight="balanced")
            else:
                LLRCV = LogisticRegressionCV(Cs=c_values, cv=cv_folds, penalty='l1', solver="saga", scoring=scoring, tol=tolerance,
                                             max_iter=max_iter)

            print("computing")
            LLRCV.fit(train_x.to_numpy(), train_y.to_numpy().reshape(train_y.to_numpy().shape[0], ))
            self.LLRCV = LLRCV

    def plot_ROC_just_curve(self, model_name, X, y):
        classifier = self.LLRCV
        color = {'Lasso logistic regression - test': 'darkgreen', 'Lasso logistic regression - train': 'darkblue'}
        probas = classifier.decision_function(X)
        print("y_true", y.to_numpy().reshape(y.shape[0],).shape, "y_prob", probas.shape)
        fpr, tpr, thresholds = roc_curve(y.to_numpy().reshape(y.shape[0],), probas)
        roc_auc = auc(fpr, tpr)
        print_to_tb(self.textBrowser,model_name + r' ROC (AUC: %0.2f)' % (roc_auc))
        plt.plot(fpr, tpr, color=color[model_name], label=model_name + r' ROC (AUC: %0.2f)' % (roc_auc), lw=2, alpha=.9)
        return None

    def plot_ROC_lasso(self):

        plt.figure(num ="Lasso logistic regression ROC"+" the "+str(self.run_index)+" run", figsize=(5, 5))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Chance', alpha=.8)
        self.plot_ROC_just_curve("Lasso logistic regression - train", self.data["train_x"], self.data["train_y"])
        if self.have_test:
            self.plot_ROC_just_curve("Lasso logistic regression - test", self.data["test_x"], self.data["test_y"])

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('1 - Specificity', fontsize=7)
        plt.ylabel('Sensitivity', fontsize=7)
        # plt.title(title +' ROC curve using test set')
        plt.legend(loc="lower right", prop={'size': 7})
        plt.show(block=False)

    def plot_ROC_multiclass(self, testing=False):
        classifier = self.LLRCV
        n_classes = len(classifier.classes_)
        if testing:
            y_score = classifier.decision_function(self.data["test_x"])
            y = label_binarize(self.data["test_y"], classes=classifier.classes_)
        else:
            y_score = classifier.decision_function(self.data["train_x"])
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
            plt.figure(num=" ROC on testing dataset"+" the "+str(self.run_index)+" run",figsize=(6,6))
        else:
            plt.figure(num=" ROC on training dataset"+" the "+str(self.run_index)+" run",figsize=(6,6))
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
            [(0.878, 0.4627, 0.3529), (0.392, 0, 0), (0.2, 0.21, 0.38), (0.4, 0.843, 0.513), (0.274, 0.51, 0.878), (0.21, 0.6627, 0.576),
            (0, 0.3568, 0.61), (0.42, 0.8588, 0.7682), (0.8, 0.43, 0.666),
             (0.59, 0.2549, 0.1)])
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

        LLRCV = self.LLRCV

        self.class_name = ["class " + str(i) for i in self.LLRCV.classes_]
        print_tb_header(text_browser=self.textBrowser, run_num=self.run_index)
        if self.multiclass:
            print(LLRCV.C_)
            for i in range(len(LLRCV.coef_)):
                coef = LLRCV.coef_[i]
                for j in range(len(coef)):
                    if coef[j] != 0:
                        print_to_tb(self.textBrowser, self.class_name[i],
                                    self.data["train_x"].columns.values[0:][j], coef[j])
                print_to_tb(self.textBrowser, self.class_name[i], "number of selected features：",len(coef[coef != 0]), "/", len(coef))
                print_to_tb(self.textBrowser, self.class_name[i], "model's C value: ", LLRCV.C_[i])
                print_to_tb(self.textBrowser, self.class_name[i], "model's interceprt: ", LLRCV.intercept_[i])

            self.plot_ROC_multiclass(False)
            if self.have_test:
                self.plot_ROC_multiclass(True)
        else:
            for i in range(len(LLRCV.coef_)):
                coef = LLRCV.coef_[i]
                for j in range(len(coef)):
                    if coef[j] != 0:
                        print_to_tb(self.textBrowser, self.data["train_x"].columns.values[0:][j], coef[j])
                print_to_tb(self.textBrowser, "number of non-zero features：", len(coef[coef != 0]), "/", len(coef))
                print_to_tb(self.textBrowser, "model's C value: ", LLRCV.C_[i])
                print_to_tb(self.textBrowser, "model's interceprt: ", LLRCV.intercept_[i])
            self.plot_ROC_lasso()


    @staticmethod
    def crack(integer):

        start = int(np.ceil(np.sqrt(integer)))
        factor = integer / start
        factor = int(np.ceil(factor))
        return (start, factor)

    def plot_regularization_path(self):

        model = self.LLRCV
        figsize = (6, 8)
        linewidth = 2
        best_c = model.C_[0]
        log_Cs = np.log10(model.Cs_)
        print("best c", best_c)
        fig, (ax1, ax2) = plt.subplots(2, 1,num=" Lasso Logstic Regression regularization profile"+" the "+str(self.run_index)+" run", figsize=figsize)

        std_score = model.scores_[1].transpose().std(axis=-1)
        mean_score = model.scores_[1].transpose().mean(axis=-1)

        higher = mean_score + std_score
        lower = mean_score - std_score

        print(log_Cs.shape, model.scores_[1].transpose().shape)


        ax1.set(ylim=[min(lower)-0.1, max(higher)+0.1])
        ax1.plot(log_Cs, mean_score, color=(0.9, 0.1, 0.1), label='Average across the folds', linewidth=linewidth)
        ax1.plot(log_Cs, higher, 'b--')
        ax1.plot(log_Cs, lower, 'b--')
        ax1.fill_between(log_Cs, higher, lower, alpha=0.2)

        ax1.axvline(np.log10(best_c), linestyle='--', color='k', label='Best C: {0}'.format(round(best_c, 4)))
        ax1.set(xlim=[min(log_Cs), max(log_Cs)])
        print('best C from CV {0}'.format(round(best_c, 4)))
        ax1.legend(prop={'size': 8}, loc="upper left")
        ax1.set_xlabel(xlabel='Log10(C)', fontsize=7)
        ax1.set_ylabel(ylabel=self.scoring_comb.currentText(), fontsize=7)
        ax1.set_title(label='Performance along the regularization path', loc="left")

        ###################### 绘图lasso coefficient的path：
        print("model.coefs_paths_", len(model.coefs_paths_), type(model.coefs_paths_))

        coefs_lasso = model.coefs_paths_[1]

        print("alphas_lasso shape", log_Cs.shape)
        print("coefs_lasso.mean(axis=0) shape", coefs_lasso.mean(axis=0).transpose().shape)

        # Display results
        colors = cycle(
            [(0.2, 0.21, 0.38), (0.4, 0.843, 0.513), (0.274, 0.51, 0.878), (0.392, 0, 0), (0.21, 0.6627, 0.576),
             (0.878, 0.4627, 0.3529), (0, 0.3568, 0.61), (0.42, 0.8588, 0.7682), (0.8, 0.43, 0.666),
             (0.59, 0.2549, 0.1)])

        for coef_l, c in zip(coefs_lasso.mean(axis=0).transpose(), colors):
            l1 = ax2.plot(log_Cs, coef_l, c=c, linewidth=linewidth * 0.85)
        print("log10 alpha mean", np.log10(best_c))

        ax2.axvline(np.log10(best_c), linestyle='--', color='k', label='Best C: {0}'.format(round(best_c, 4)))
        ax2.set(xlim=[min(log_Cs), max(log_Cs)])
        ax2.set_xlabel(xlabel='Log10(C)', fontsize=7)
        ax2.set_ylabel(ylabel='Coefficients', fontsize=7)
        ax2.set_title(label="Coefficients and Intercept Path", loc="left")
        ax2.legend(prop={'size': 8}, loc="upper left")
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)
        plt.show(block=False)
        return None

    def plot_regularization_path_multiclass(self):
        model = self.LLRCV
        linewidth = 2
        log_Cs = np.log10(model.Cs_)
        row_num, col_num = self.crack(len(model.classes_) + 1)
        figsize = (5 * row_num, 4 * col_num)
        fig, axs = plt.subplots(row_num, col_num, num=" Lasso Logstic Regression regularization profile"+" the "+str(self.run_index)+" run", figsize=figsize)

        std_score = model.scores_[1].transpose().std(axis=-1)
        mean_score = model.scores_[1].transpose().mean(axis=-1)

        higher = mean_score + std_score
        lower = mean_score - std_score

        print(log_Cs.shape, model.scores_[1].transpose().shape)

        axs = axs.reshape(1,axs.shape[0] + axs.shape[1])

        axs[0, 0].set(ylim=[min(lower) - 0.1, max(higher) + 0.1])
        axs[0, 0].plot(log_Cs, mean_score, color=(0.9, 0.1, 0.1), label='Average across the folds', linewidth=linewidth)
        axs[0, 0].plot(log_Cs, higher, 'b--')
        axs[0, 0].plot(log_Cs, lower, 'b--')
        axs[0, 0].fill_between(log_Cs, higher, lower, alpha=0.2)
        if model.C_[0] == model.C_[1] == model.C_[2]:
            axs[0, 0].axvline(np.log10(model.C_[0]), linestyle='--', color='k',
                          label='Best C: {0}'.format(round(model.C_[0], 4)))
        axs[0, 0].set(xlim=[min(log_Cs), max(log_Cs)])
        axs[0, 0].legend(prop={'size': 8}, loc="upper left")
        axs[0, 0].set_xlabel(xlabel='Log10(C)', fontsize=7)
        axs[0, 0].set_ylabel(ylabel=self.scoring_comb.currentText(), fontsize=7)
        axs[0, 0].set_title(label='Performance along the regularization path', loc="left")

        ###################### 绘图lasso的path：
        print("model.coefs_paths_", len(model.coefs_paths_), type(model.coefs_paths_))
        for i in range(1,len(model.classes_)+1):
            coefs_lasso = model.coefs_paths_[model.classes_[i-1]]
            best_c = model.C_[i-1]
            print("alphas_lasso shape", log_Cs.shape)
            print("coefs_lasso.mean(axis=0) shape", coefs_lasso.mean(axis=0).transpose().shape)

            # Display results
            colors = cycle(
                [(0.2, 0.21, 0.38), (0.4, 0.843, 0.513), (0.274, 0.51, 0.878), (0.392, 0, 0), (0.21, 0.6627, 0.576),
                 (0.878, 0.4627, 0.3529), (0, 0.3568, 0.61), (0.42, 0.8588, 0.7682), (0.8, 0.43, 0.666),
                 (0.59, 0.2549, 0.1)])

            for coef_l, c in zip(coefs_lasso.mean(axis=0).transpose(), colors):
                l1 = axs[0, i].plot(log_Cs, coef_l, c=c, linewidth=linewidth * 0.85)
            print("log10 alpha mean", np.log10(best_c))

            axs[0, i].axvline(np.log10(best_c), linestyle='--', color='k',
                              label='Best C: {0}'.format(round(best_c, 4)))
            axs[0, i].set(xlim=[min(log_Cs), max(log_Cs)])
            axs[0, i].set_xlabel(xlabel='Log10(C)', fontsize=7)
            axs[0, i].set_ylabel(ylabel='Coefficients', fontsize=7)
            axs[0, i].set_title(label='class '+ str(i-1) + " coefficients path", loc="left")
            axs[0, i].legend(prop={'size': 8}, loc="upper left")
            plt.tight_layout(pad=1, w_pad=1, h_pad=1)
            plt.show(block=False)
        return None

    def print_classification_report(self):

        if self.LLRCV is None:
            return
        print_classification_report_public(model=self.LLRCV, data=self.data, class_name=self.class_name, textBrowser=self.textBrowser, have_test=self.have_test)


    def plot_confusion_matrix(self):
        plot_confusion_matrix_public(model=self.LLRCV, model_name="Lasso Logistic Regression", data=self.data, class_name=self.class_name, run_index=self.run_index, have_test=self.have_test)

    def transform_data(self):
        sfm = SelectFromModel(estimator=self.LLRCV, prefit=True)
        self.data["train_x"] = pd.DataFrame(sfm.transform(self.data["train_x"]), index=self.data["train_x"].index, columns=self.data["train_x"].columns[sfm.get_support()])
        self.transformed = True
        print_to_tb(self.textBrowser, "number of finially selected features：", len(sfm.get_support()[sfm.get_support() == True]), "/", len(self.LLRCV.coef_[0]))
        print_log_header(self.parentWidget().class_name)
        print_to_log("train_x" + str(self.data["train_x"].shape))
        print_to_log("train_y" + str(self.data["train_y"].shape))
        if self.have_test:
            self.data["test_x"] = pd.DataFrame(sfm.transform(self.data["test_x"]), index=self.data["test_x"].index, columns=self.data["test_x"].columns[sfm.get_support()])
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
            self.train_a_model()

        if check_data_model_compatible_public(self.data, self.LLRCV, "coef_", cv=False):
            return None

        self.print_result()

        if self.plot_lasso_cb.checkState():
            if len(self.LLRCV.classes_)>2:
                self.plot_regularization_path_multiclass()
            else:
                self.plot_regularization_path()

        if self.output_cla_rep_cb.checkState():
            self.print_classification_report()

        if self.output_confusion_cb.checkState():
            self.plot_confusion_matrix()

        if self.save_model_cb.checkState() and self.save_model_later:
            self.save_a_model()

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
    Dialog = Lasso_Logistic_Regression_Dialog()
    Dialog.show()
    sys.exit(app.exec_())
