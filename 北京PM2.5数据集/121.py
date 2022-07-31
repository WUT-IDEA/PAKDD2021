import pandas as pd
import numpy as np



# 定义计算熵的函数
def ent(data1):
    prob1 = pd.value_counts(data1) / len(data1)
    epsilon = 1e-5
    return sum(np.log2(prob1+epsilon) * prob1 * (-1))


# 定义计算信息增益的函数
def gain(data, str1, str2):
    e1 = data.groupby(str1).apply(lambda x: ent(x[str2]))
    p1 = pd.value_counts(data[str1]) / len(data[str1])
    e2 = sum(e1 * p1)
    return ent(data[str2]) - e2

if __name__ == '__main__':
    data = pd.read_csv("D:\桌面\空气质量预测综合\北京PM2.5数据集\多步预测\data\data_1009.csv")
    data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'])
    data.set_index("Unnamed: 0", inplace=True)

    # 取一个月的数据
    oneMonth = data[data.index >= "2014-12-01"]
    oneMonth.rename(columns={"pm2.5": "pm2_5"}, inplace=True)
    oneMonth["pm2_5_cut"] = pd.cut(oneMonth.pm2_5, bins=[1, 10, 20, 30, 40, 100, 200, 300, 400, 500])
    oneMonth["TEMP_cut"] = pd.cut(oneMonth.TEMP, bins=[-10, -6, -2, 2, 6, 10, 15])
    print(gain(oneMonth,'TEMP_cut','pm2_5_cut'))