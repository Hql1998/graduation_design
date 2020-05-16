import pandas as pd

def trnsform_nn_to_three_columns(df):
    # pd.DataFrame.from_dict({"A":["a","a","a"],"B":["c","b","d"],"Value":[0,1,2]})
    result_dict = {"A":[], "B":[], "Value":[]}
    for row_index_slow in range(0, df.shape[0]):
        A = df.index[row_index_slow]
        for row_index_fast in range(row_index_slow+1, df.shape[0]):
            B = df.index[row_index_fast]
            value = df.iloc[row_index_fast,row_index_slow]
            result_dict["A"].append(A)
            result_dict["B"].append(B)
            result_dict["Value"].append(value)
    result_data = pd.DataFrame.from_dict(result_dict)
    return result_data

#输入的矩阵应该是严格n*n的方阵，并且第一行和第一列是名称，顺序一一对应。专门导入cytospace
df = pd.read_excel(r"E:\课件\yanghuixuejie\gaoying\gene_chemical_jz.xlsx", index_col=0)
result_data = trnsform_nn_to_three_columns(df)
result_data.to_excel(r"E:\课件\yanghuixuejie\gaoying\gene_chemical_jz_result.xlsx")
