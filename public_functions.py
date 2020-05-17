from sklearn.metrics import roc_curve, auc, classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from pandas import read_excel,read_csv
from print_to_log import *
from PyQt5.Qt import QErrorMessage, qErrnoWarning


def plot_confusion_matrix_public(model=None, model_name=None, data=None, class_name=None, run_index=1, have_test=False):

    train_disp = plot_confusion_matrix(model, data["train_x"], data["train_y"],
                                       display_labels=class_name, values_format="d",
                                       cmap=plt.cm.Oranges)
    train_disp.figure_.canvas.set_window_title(
        'confusion matrix on training set' + " the " + str(run_index) + " run " + model_name)
    train_disp.ax_.set_title("confusion matrix on training set")
    if have_test:
        test_disp = plot_confusion_matrix(model, data["test_x"], data["test_y"],
                                          display_labels=class_name, values_format="d",
                                          cmap=plt.cm.Oranges)
        test_disp.figure_.canvas.set_window_title(
            'confusion matrix on testing set' + " the " + str(run_index) + " run " + model_name)
        test_disp.ax_.set_title("confusion matrix on testing set")

    plt.show(block=False)

def print_classification_report_public(model,data,class_name,textBrowser,have_test):

        train_y = data["train_y"]
        train_y_pre = model.predict(data["train_x"])
        train_resut_string = classification_report(train_y, train_y_pre, target_names=class_name)
        print_to_tb(textBrowser, "*"*10 + "\ntraining set: \n" + train_resut_string)
        if have_test:
            test_y = data["test_y"]
            test_y_pre = model.predict(data["test_x"])
            test_resut_string = classification_report(test_y, test_y_pre, target_names=class_name)
            print_to_tb(textBrowser, "*" * 10 + "\ntesting set: \n" + test_resut_string)

def save_file_public(open_result, data, have_test, widget_name, run_index):
    file_path = open_result[0]
    if open_result[1] == "csv(*.csv)":
        data["train_x"].merge(data["train_y"], left_index=True, right_index=True).to_csv(
            file_path.replace(".csv", "_train_{0}{1}.csv".format(widget_name, "_the_" + str(run_index) + "_run")))
        if have_test:
            data["test_x"].merge(data["test_y"], left_index=True, right_index=True).to_csv(
                file_path.replace(".csv", "_test_{0}{1}.csv".format(widget_name,
                                                                    "_the_" + str(run_index) + "_run")))
    elif open_result[1] == "excel(*.xlsx)":
        data["train_x"].merge(data["train_y"], left_index=True, right_index=True).to_excel(
            file_path.replace(".xlsx", "_train_{0}{1}.xlsx".format(widget_name,
                                                                   "_the_" + str(run_index) + "_run")))
        if have_test:
            data["test_x"].merge(data["test_y"], left_index=True, right_index=True).to_excel(
                file_path.replace(".xlsx", "_test_{0}{1}.xlsx".format(widget_name,
                                                                      "_the_" + str(run_index) + "_run")))

def check_data_public(data):
    have_test = True
    no_y_flag = False
    if data["test_y"] is None:
        have_test = False
    if data["train_y"] is None:
        no_y_flag = True
        QErrorMessage.qtHandler()
        qErrnoWarning("you didn't have a training target label")
    return no_y_flag, have_test

def check_data_model_compatible_public(data,model,check_string,cv=False):
    if model is None:
        QErrorMessage.qtHandler()
        qErrnoWarning("you didn't have a fitted model")
        return True
    if data["train_x"] is None:
        QErrorMessage.qtHandler()
        qErrnoWarning("you didn't have training data")
        return True

    if cv:
        model = model.best_estimator_

    if check_string =="n_features_":
        try:
            if model.n_features_ != data["train_x"].shape[1]:
                QErrorMessage.qtHandler()
                qErrnoWarning("the data didn't fit the model, n_features_, the model assumed {0} features, while input {1} features".format(model.n_features_, data["train_x"].shape[1]))
                return True
        except:
            QErrorMessage.qtHandler()
            qErrnoWarning("counting feature method didn't fit, n_features_, probably you load a wrong model")
            return True
    elif check_string =="coef_":
        try:
            if model.coef_.shape[1] != data["train_x"].shape[1]:
                QErrorMessage.qtHandler()
                qErrnoWarning("the data didn't fit the model, coef_, the model assumed {0} features, while input {1} features".format(model.coef_.shape[1], data["train_x"].shape[1]))
                return True
        except:
            QErrorMessage.qtHandler()
            qErrnoWarning("counting feature method didn't fit, coef_, probably you load a wrong model")
            return True
    elif check_string == "shape_fit_":
        try:
            if model.shape_fit_[1] != data["train_x"].shape[1]:
                QErrorMessage.qtHandler()
                qErrnoWarning("the data didn't fit the model, coef_, the model assumed {0} features, while input {1} features".format(model.coef_.shape[1], data["train_x"].shape[1]))
                return True
        except:
            QErrorMessage.qtHandler()
            qErrnoWarning("counting feature method didn't fit, coef_, probably you load a wrong model")
            return True

    else:
        QErrorMessage.qtHandler()
        qErrnoWarning("the model don't have a counting feature method")
        return True

    return False

def read_data_public(data):
    data["train_x"] = read_excel(r"E:\python\graduation_design\temp\cleaned_diabetes_binary_train_data_preprocessing.xlsx", index_col=0).drop(columns="class")
    # data["train_x"] = read_csv(r"E:\python\graduation_design\temp\balanced_diabetes_dataset.csv").drop(columns="class")
    data["test_x"] = read_excel(r"E:\python\graduation_design\temp\cleaned_diabetes_binary_test_data_preprocessing.xlsx", index_col=0).drop(columns="class")
    data["train_y"] = read_excel(
        r"E:\python\graduation_design\temp\cleaned_diabetes_binary_train_data_preprocessing.xlsx", index_col=0).loc[:,["class"]]
    # data["train_y"] = read_csv(r"E:\python\graduation_design\temp\balanced_diabetes_dataset.csv").loc[:,["class"]]
    data["test_y"] = read_excel(
        r"E:\python\graduation_design\temp\cleaned_diabetes_binary_test_data_preprocessing.xlsx", index_col=0).loc[:,["class"]]
    print(data["train_x"].info(verbose=True))

def read_data_multiclass_public(data):

    data["train_x"] = read_excel(r"E:\python\graduation_design\temp\cleaned_radio_multiclass_train_data_preprocessing.xlsx", index_col=0).drop(columns="mutation_0forNo_1for19_2forL858R")
    data["test_x"] = read_excel(r"E:\python\graduation_design\temp\cleaned_radio_multiclass_test_data_preprocessing.xlsx", index_col=0).drop(columns="mutation_0forNo_1for19_2forL858R")
    data["train_y"] = read_excel(
        r"E:\python\graduation_design\temp\cleaned_radio_multiclass_train_data_preprocessing.xlsx", index_col=0).loc[:,["mutation_0forNo_1for19_2forL858R"]]
    data["test_y"] = read_excel(
        r"E:\python\graduation_design\temp\cleaned_radio_multiclass_test_data_preprocessing.xlsx", index_col=0).loc[:,["mutation_0forNo_1for19_2forL858R"]]
    print(data["train_x"].info(verbose=True))


