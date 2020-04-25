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
strat_train_set,strat_test_set = splite_train_test_stratified(housing,obj_col="median_income",bin_number=6,test_ratio=0.2)

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