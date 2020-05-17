import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import label_binarize

data1 = np.array([[0.2, 0.3, 0.8],[0.7, 0.1, 0.1],[0.2, 0.9, 0.6]])
print(data1)
print()

data2 = np.array([[0.3, 0.3, 0.8],[0.8, 0.2, 0.1],[0.3, 0.9, 0.6]])
print(data2)
print()

data3 = np.array([[0.1, 0.3, 0.8],[0.6, 0.3, 0.1],[0.1, 0.9, 0.6]])
print(data3)
print()

data = [data1,data2,data3]
data_mean = np.array(data1)
for i in range(data1.shape[1]):
    data_mean[:,i] = np.array([j[:,i] for j in data]).mean(axis=0)
print(data_mean)
classes_ = ["class1", "class2", "class3"]
classes_[i]
print([np.argmax(i) for i in data_mean[:,:]])
