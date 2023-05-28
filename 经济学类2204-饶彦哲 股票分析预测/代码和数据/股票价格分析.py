#  经济学类2204 饶彦哲  0122215970408  2023-5-22
# 数据顺序是 date open high low close adj close volume
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split  # 交叉验证 #将数据分为测试集和训练集
from sklearn import preprocessing  # 数据标准化
from sklearn.linear_model import LinearRegression  # 线性回归
from fbprophet import Prophet
import mplfinance as mpf

sns.set_theme()  # 使用seaborn的默认主题


def read(filename):
    # ##   文件读取为dataframe
    df = pd.read_csv(filename)
    return df


def plt_show(x_ls, y_ls):
    plt.plot(x_ls, y_ls)
    plt.show()


def sns_show(x_ls, y_ls, kinds):
    sns.relplot(x=x_ls, y=y_ls, kind=kinds)
    plt.show()


def K_line(df, t):
    # ##   绘制k线图

    df = df[t:]  # ##   绘制近期k线
    df.index = pd.DatetimeIndex(df['Date'])
    mpf.plot(df, type='candle', mav=(2, 5, 10), volume=True, style='charles')


def relation_analysis(df1, df2):
    # ##   相关性分析

    dfcomp = pd.DataFrame()
    dfcomp['Close1'] = df1['Adj Close']
    dfcomp['Close2'] = df2['Adj Close']
    # ##   用调整后的收盘价来验证两只股票相关性

    retscomp = dfcomp.pct_change()
    corr = retscomp.corr()

    # plt.scatter(retscomp.Close1, retscomp.Close2)
    # plt.xlabel('jp')
    # plt.ylabel('gm')
    # plt.show()
    # ##  绘制皮尔森相关系数散点图

    scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10))
    plt.show()
    ##  绘制两只股票收盘价的散点矩阵图

    # sns.heatmap(df1.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)
    # plt.show()
    # ## 绘制第一支股票内部数据相关性的热力图

    # sns.heatmap(dfcomp.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)
    # plt.show()
    # ##  绘制两只股票价格相关性的热力图


def create_data(df):
    # ## 数据处理

    df['Change'] = (df['Close'] - df['Open']) / df['Open']
    df['XChange'] = (df['High'] - df['Low']) / df['Low']
    df = df[['Close', 'Change', 'XChange', 'Volume']]
    return df


def analysis(x, y):
    # ## 使用 sklearn 根据其他数据来预测 Adj Close

    x = skl.preprocessing.scale(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    res = LinearRegression()
    # ## 使用线性回归

    res.fit(x_train, y_train)
    s = res.predict(x_test)
    accuracy = res.score(x_test, y_test)

    print(accuracy)


def predict(df):
    # ## 使用 prophet 来预测未来一年的股票价格走势

    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = np.log(df['Close'])
    df['ds'] = df['Date']
    df['y'] = df['Close']
    df = df[['ds', 'y']]
    df_train = df[:4500]
    df_test = df[4500:]

    model = Prophet(daily_seasonality=True)
    model.fit(df_train)
    future = model.make_future_dataframe(periods=365, freq='D')
    forecast = model.predict(future)
    # ## 模型训练

    model.plot(forecast)
    plt.show()
    model.plot_components(forecast)
    # ## 输出时序预测图

    df_test = df_test.set_index('ds')
    forecast = forecast[['ds', 'yhat']].set_index('ds')
    df_test['y'] = np.exp(df_test['y'])
    forecast['yhat'] = np.exp(forecast['yhat'])
    df_all = forecast.join(df_test).dropna()
    df_all.plot()
    plt.legend(['true', 'yhat'])
    plt.show()
    # ##   预测值和真实值进行对比


if __name__ == '__main__':
    jpmorgan = read("datasets\\JPMorgan Chase.csv")
    goldman = read("datasets\\The Goldman Sachs.csv")
    ls = ['jpmorgan', 'goldman']  # ##  创建股票名称列表，方便添加更多股票统一分析

    # n = int(input())
    n = 5800
    # ##  查看指定时间之后的K线图
    K_line(jpmorgan, n)
    K_line(goldman, n)

    relation_analysis(jpmorgan, goldman)

    jp = create_data(jpmorgan)
    gm = create_data(goldman)

    analysis(jp, jpmorgan['Adj Close'])
    analysis(gm, goldman['Adj Close'])

    predict(jpmorgan)
    predict(goldman)
    # ##   第三方库实在是过于简洁了，很多功能几行代码就写完了，所以代码量确实不大。。
