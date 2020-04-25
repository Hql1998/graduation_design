import pandas as pd
import numpy
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder

df = pd.DataFrame([["a","1"],["b","2"],["b","0"],["a","2"],["c","3"]],columns=["zimu","shuzi"])

print(df.isnull().values.sum())
df = df.iloc[:,[1]]

oe = OneHotEncoder(sparse=False)

df_encoded = oe.fit_transform(df.to_numpy())#.toarray()

for i,v in enumerate(df.columns):
    for j in oe.categories_[i]:
        print(v+"_"+j)

# dataFrame = pd.read_excel("./temp/radiomic_test_data.xlsx")
# simple_imputer = SimpleImputer(strategy="median")
# data_numpy = simple_imputer.fit_transform(dataFrame.to_numpy())
# list(dataFrame.dtypes[dataFrame.dtypes == numpy.object].to_dict().keys())
# print(dataFrame.to_numpy())