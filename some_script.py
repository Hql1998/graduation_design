from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

##使用多种方法划分训练集和测试集

## 自定义随机分割函数
def splite_train_test(data, test_ratio = 0.2):
    """the function will return train set and test set
    """
    shuffled_index = np.random.permutation(len(data))
    test_set_size = int(test_ratio * len(data))
    test_indeces = shuffled_index[:test_set_size]
    train_indecs = shuffled_index[test_set_size:]
    return data.iloc[train_indecs],data.iloc[test_indeces]


def splite_train_test_stratified(data,obj_col,bin_number=6,test_ratio=0.2):
    # 需要后序修改
    data["income_cat"] = pd.cut(data[obj_col],bins=[i for i in np.arange(0,bin_number+0.1,1.5)] + [np.inf],labels=range(1,6))
    SSsplit = StratifiedShuffleSplit(n_splits=1,test_size=test_ratio,random_state=42)
    for train_index, test_index in SSsplit.split(housing,housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    return strat_train_set,strat_test_set

## 自己设置的随机分配函数
train_set, test_set = splite_train_test(housing, test_ratio = 0.2)

## 使用sklearn包的随机分配函数
# train_set, test_set = train_test_split(housing,test_size=0.2, random_state=42)

## 使用分层分割的方法
strat_train_set,strat_test_set = splite_train_test_stratified(housing, obj_col="median_income",bin_number=6,test_ratio=0.2)

print(strat_train_set.loc[:,"income_cat"].value_counts()/len(strat_train_set))
# print(housing.loc[:,"income_cat"].value_counts()/len(housing))
# print(train_set.loc[:,"income_cat"].value_counts()/len(train_set))

# print(strat_train_set.loc[:,"income_cat"].value_counts())
# print(housing.loc[:,"income_cat"].value_counts())
# print(train_set.loc[:,"income_cat"].value_counts())
for set_ in (strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1,inplace=True)



import matplotlib.pyplot as plt
from IPython.display import display, HTML

housing = strat_train_set.copy()
housing.head()
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.3,
             s=housing["population"]/100,label="Population",
             figsize=(10,7),
             c="median_house_value",colormap=plt.get_cmap("jet"),colorbar=True)
plt.show()

corr_matrix = housing.corr()
display(corr_matrix.sort_values(by="median_house_value",ascending=False))

from pandas.plotting import scatter_matrix

attributes = ['median_house_value','median_income','housing_median_age']
scatter_matrix(housing[attributes],figsize=(12,8))

housing.plot(kind="scatter",x="median_income",y="median_house_value")


# 转换数据，计算出有用的数据值：
housing["room_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedroom_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
display(corr_matrix["median_house_value"].sort_values(ascending=False))


# 数据预处理，处理catagory类型的数据：

# 分离X与y
housing = strat_train_set.drop("median_house_value",axis=1)
housing_label = strat_train_set["median_house_value"].copy()

# df["col_name"] -> pandas.core.series.Series
# df[["col_name"]] -> pandas.core.frame.DataFrame

housing_cat = housing[["ocean_proximity"]]

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder

ordinal_encoder = OrdinalEncoder()
# 只接受dataFrame格式不接受Series格式的
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

ordinal_encoder.categories_
cat_encoder.categories_

housing_cat_1hot.toarray()

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer,MissingIndicator

numeric_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("attribute_adder",CombinedAttributesAdder()),
    ("standarization",StandardScaler())
])

housing_num = housing.drop("ocean_proximity",axis = 1)

housing_num_tr = numeric_pipeline.fit_transform(housing_num)
# display(housing)
# display(housing_num)



###_____________________________________________________________


# 通过LassoCV: coordinate descent选择最优的lambda值（在python里lambda值为alpha）
def lassoData(X0, X_test0, y, y_test):
    EPSILON = 1e-6
    scaler = StandardScaler()
    scaler.fit(X0)
    X = scaler.transform(X0)
    scaler.fit(X_test0)
    X_test = scaler.transform(X_test0)

    alphas = []
    feature_number = 0
    iter_times = 1
    for i in range(0, iter_times):
        print("Computing regularization path using the coordinate descent lasso...")
        model_0 = LassoCV(cv=10, eps=EPSILON, max_iter=5000, normalize=False, random_state=i * 2 + 10).fit(X,
                                                                                                           y)  # ,tol=0.0001
        alphas.append(model_0.alpha_)
        print("系数不为0的特征个数为：", len(model_0.coef_[model_0.coef_ != 0]))
        feature_number += len(model_0.coef_[model_0.coef_ != 0])

        # Set a minimum threshold of 1e-18
    print(np.mean(alphas))
    #     alpha_mean = 0.023061745220722834
    alpha_mean = np.mean(alphas)
    if feature_number / iter_times >= 14:
        alpha_mean = 2 * alpha_mean
    elif feature_number / iter_times > 12:
        alpha_mean = 1.2 * alpha_mean
    elif feature_number / iter_times < 5:
        alpha_mean = 0.7 * alpha_mean
    print("really alpha mean: ", alpha_mean)

    model = Lasso(alpha=alpha_mean, max_iter=6000).fit(X, y)
    sfm = SelectFromModel(model, prefit=True, threshold=1e-18)
    n_features = sfm.transform(X)  # .shape[1]
    n_features_test = sfm.transform(X_test)

    #     scaler = StandardScaler()
    #     scaler.fit(n_features)
    #     n_features = scaler.transform(n_features)
    print("最终特征数量", np.sum(model.coef_ != 0))
    feature_list = []
    for i in range(len(model.coef_)):
        if (model.coef_[i] != 0):
            print("系数不为0的特征为：", X0.columns.values[0:][i], model.coef_[i])
            feature_list.append(X0.columns.values[0:][i])
    print("interceprt:", model.intercept_)
    X_lasso = model.predict(X)
    X_lasso_test = model.predict(X_test)

    #     print(n_features,feature_list,sep="\n")
    tem_lasso_result = pd.DataFrame(n_features, columns=feature_list)
    tem_lasso_predic = pd.DataFrame(np.array([X_lasso, y]).transpose(), columns=["prediction", "mutation"])
    tem_lasso = pd.merge(tem_lasso_result, tem_lasso_predic, left_index=True, right_index=True)
    tem_lasso.to_csv(lasso_temp_result)

    tem_lasso_result_test = pd.DataFrame(n_features_test, columns=feature_list)
    tem_lasso_predic_test = pd.DataFrame(np.array([X_lasso_test, y_test]).transpose(),
                                         columns=["prediction", "mutation"])
    tem_lasso_test = pd.merge(tem_lasso_result_test, tem_lasso_predic_test, left_index=True, right_index=True)
    tem_lasso_test.to_csv(lasso_temp_result_test)

    return [X_lasso, X_lasso_test, ["image_signature"], model, model_0, alpha_mean]


def lasso_alpha_fig(X0, X_test0, y, y_test, model_0, alpha_mean, figsize):
    EPSILON = 1e-6
    scaler = StandardScaler()
    scaler.fit(X0)
    X = scaler.transform(X0)
    scaler.fit(X_test0)
    X_test = scaler.transform(X_test0)
    #     ## draw pics
    #     # Display results
    m_log_alphas = np.log10(model_0.alphas_ + EPSILON)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    #     plt.figure(figsize=(3.15,2.8))
    print(m_log_alphas.shape, model_0.mse_path_.shape)
    ax1.plot(m_log_alphas, model_0.mse_path_, ':')
    ax1.plot(m_log_alphas, model_0.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=linewidth)
    ax1.axvline(np.log10(alpha_mean + EPSILON), linestyle='--', color='k',
                label='alpha: CV estimate {0}'.format(round(alpha_mean, 4)))
    ax1.set(xlim=[-6.8, 0])
    print('coordinate descent alpha CV {0}'.format(round(alpha_mean, 4)))
    #     ax1.legend(prop={'size': 4.5},loc=(0.24,0.84))
    ax1.legend(prop={'size': 4.5}, loc="upper center")
    ax1.set_xlabel(xlabel='Log(lambda)', fontsize=7)
    ax1.set_ylabel(ylabel='Mean square error', fontsize=7)
    ax1.set_title(label='a', loc="left", fontdict=fontdic)
    #     plt.axis('tight')
    #     if pre_19_21:
    #         plt.savefig(fig_19_21_lasso_mse,dpi=300,quality=95,bbox_inches = "tight")
    #     else:
    #         plt.savefig(fig_mutation_lasso_mse,dpi=300,quality=95,bbox_inches = "tight")
    #     plt.show()

    ###################### 绘图lasso的path：
    from itertools import cycle
    from sklearn.linear_model import lasso_path

    eps = EPSILON  # the smaller it is the longer is the path
    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=True, alphas=model_0.alphas_, normalize=False)

    # Display results
    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_lasso = np.log10(alphas_lasso)
    for coef_l, c in zip(coefs_lasso, colors):
        l1 = ax2.plot(neg_log_alphas_lasso, coef_l, c=c)
    ax2.axvline(np.log10(alpha_mean + EPSILON), linestyle='--', color='k', label='alpha: {0}'.format(alpha_mean))
    ax2.set(xlim=[-6.8, 0])
    ax2.set_xlabel(xlabel='Log(lambda)', fontsize=7)
    ax2.set_ylabel(ylabel='coefficients', fontsize=7)
    ax2.set_title(label='b', loc="left", fontdict=fontdic)

    #     if pre_19_21:
    #         ax2.text(1.5, 2, 'alpha: {0}'.format(round(alpha_mean,4)),fontsize=7)
    #     else:
    #         ax2.text(2, 4, 'alpha: {0}'.format(round(alpha_mean,4)),fontsize=7)

    #     plt.legend(l1[-1], 'Lasso', loc='lower left')
    #     plt.axis('tight')
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)

    if pre_19_21:
        plt.savefig(fig_19_21_lasso_path.replace(".", str(figsize[0]) + "."), dpi=300, quality=95, bbox_inches="tight")
    else:
        plt.savefig(fig_mutation_lasso_path.replace(".", str(figsize[0]) + "."), dpi=300, quality=95,
                    bbox_inches="tight")
    plt.show()
    return None
