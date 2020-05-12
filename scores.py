import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# data = pd.read_excel(r"E:\Project\Lisu\database_score1.xlsx", header=None,names=range(1,10))
# print(data.std(axis=0))

data = pd.read_excel(r"E:\Project\Lisu\score_correlation1.xlsx")

print(data.head())
print(data.corr())
# data.plot.scatter(x=0,y=1)

plt.scatter(data.iloc[:,0],data.iloc[:,1])
plt.xlabel("实验成绩")
plt.ylabel("总分")
plt.xlim(0,30)
plt.ylim(40,100)
plt.show()


# 箱线图
# boxplot = data.boxplot(return_type="dict", grid=False, patch_artist=True)
# colors = [(0.45, 0.893, 0.563),(0.4, 0.843, 0.513),(0.3, 0.743, 0.413),
#           (0, 0.4668, 0.75),(0, 0.4068, 0.66),(0, 0.3068, 0.56),
#           (0.652, 0, 0),(0.52, 0, 0),(0.3542, 0, 0)]
# for patch, color in zip(boxplot["boxes"], colors):
#     patch.set_facecolor(color)
#
# plt.ylabel("Score")
# plt.xlabel("year")
# plt.show()



# 堆叠图
# his = []
# weights = []
# for i,col in enumerate(data.columns):
#     print(len(data[col].dropna()))
#     his.append(data[col].dropna())
#     weights.append([1/len(data[col].dropna()) for i in range(len(data[col].dropna()))])
#     # data[col].dropna().plot.bar(x='Score', y='Num', ax=ax, stacked=True,
#     #           color=colors[i], label=col)
#
# hischar = plt.hist(his, bins=6, range=(40,100), weights=weights,  color=colors, label=[str(i) for i in data.columns])#stacked=True
# # hischar = plt.hist(his, bins=6, range=(40,100), weights=weights, stacked=True, color=colors, label=[str(i) for i in data.columns])
#
#
# print(hischar[0])
# # for i in range(len(hischar[0])):
# print(hischar[0][1,:].sum())
# plt.legend()
# plt.ylabel("Frequency")
# plt.xlabel("Scores")
# plt.show()