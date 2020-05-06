import pandas as pd
import numpy
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import label_binarize

# train_y = pd.read_excel(r"E:\python\graduation_design\temp\radio_train_data_preprocessing.xlsx").loc[:,["mutation_0forNo_1for19_2forL858R"]]
# print(train_y)
# y = label_binarize(train_y, classes=[1, 0, 2])
# print(y)

# from itertools import cycle
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import lasso_path, enet_path
# from sklearn import datasets
#
# X, y = datasets.load_diabetes(return_X_y=True)
#
# X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)
# print("X' shape", X.shape)
#
# eps = 5e-3  # the smaller it is the longer is the path
#
# print("Computing regularization path using the lasso...")
# alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)
# print("alphas_lasso shape", alphas_lasso.shape)
# print("coefs_lasso shape", coefs_lasso.transpose().shape)
#
# plt.figure(1)
# colors = cycle(['b', 'r', 'g', 'c', 'k'])
# neg_log_alphas_lasso = -np.log10(alphas_lasso)
# for coef_l, c in zip(coefs_lasso, colors):
#     l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
#
# plt.xlabel('-Log(alpha)')
# plt.ylabel('coefficients')
# plt.title('Lasso and Elastic-Net Paths')
# plt.legend('Lasso', loc='lower left')
# plt.axis('tight')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn import datasets
# from sklearn.linear_model import LassoCV
# from sklearn.linear_model import Lasso
# from sklearn.model_selection import KFold
# from sklearn.model_selection import GridSearchCV
#
# X, y = datasets.load_diabetes(return_X_y=True)
# X = X[:150]
# y = y[:150]
#
# lasso = Lasso(random_state=0, max_iter=10000)
# alphas = np.logspace(-4, -0.5, 30)
#
# tuned_parameters = [{'alpha': alphas}]
# n_folds = 5
#
# clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
# clf.fit(X, y)
# scores = clf.cv_results_['mean_test_score']
# scores_std = clf.cv_results_['std_test_score']
# plt.figure().set_size_inches(8, 6)
# plt.plot(np.log10(alphas), scores)
#
# # plot error lines showing +/- std. errors of the scores
# std_error = scores_std / np.sqrt(n_folds)
#
# plt.plot(np.log10(alphas), scores + std_error, 'b--')
# plt.plot(np.log10(alphas), scores - std_error, 'b--')
#
# # alpha=0.2 controls the translucency of the fill color
# plt.fill_between(np.log10(alphas), scores + std_error, scores - std_error, alpha=0.2)
#
# plt.ylabel('CV score +/- std error')
# plt.xlabel('alpha')
# plt.xlim([np.log10(alphas[0]), np.log10(alphas[-1])])
# plt.show()

import numpy as np
# def crack(integer):
#     start = int(np.sqrt(integer))
#     factor = integer / start
#     while not is_integer(factor):
#         start += 1
#         factor = integer / start
#     return int(factor), start
#
#
# def is_integer(number):
#     if int(number) == number:
#         return True
#     else:
#         return False





print(crack(2))