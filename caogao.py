import pandas as pd
import numpy
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder

# df = pd.DataFrame([["a","1"],["b","2"],["b","0"],["a","2"],["c","3"]],columns=["zimu","shuzi"])
#
# print(df.isnull().values.sum())
# df = df.iloc[:,[1]]
#
# oe = OneHotEncoder(sparse=False)
#
# df_encoded = oe.fit_transform(df.to_numpy())#.toarray()
#
# for i,v in enumerate(df.columns):
#     for j in oe.categories_[i]:
#         print(v+"_"+j)

dataFrame = pd.read_excel("./temp/radiomic_test_data.xlsx")

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
train_set, test_set = train_test_split(dataFrame,test_size=0.2, random_state=42)
print(type(test_set))

SSsplit = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=42)
for train_index, test_index in SSsplit.split(dataFrame ,dataFrame.iloc[:,-1]):
    strat_train_set = dataFrame.loc[train_index]
    strat_test_set = dataFrame.loc[test_index]
print(type(strat_test_set))
# simple_imputer = SimpleImputer(strategy="median")
# data_numpy = simple_imputer.fit_transform(dataFrame.to_numpy())
# list(dataFrame.dtypes[dataFrame.dtypes == numpy.object].to_dict().keys())
# print(dataFrame.to_numpy())