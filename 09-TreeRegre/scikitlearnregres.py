本节我们依旧来使用scikit-learn库来实现一下我们的树回归算法。在sklearn中，决策树回归使用的是CART算法，也就是我们所讲的分类回归树算法。
有官方图为证：
我们先来了解一下决策树回归的接口信息

参数说明：
    criterion:字符型，可选，默认为mse，即均方误差。用来衡量划分质量，也可以选择mae，即平均绝对误差。
    splitter：字符型，可选，默认为best。用于在每个节点拆分的策略选择。best即为选择最佳划分处，random即为随机选择。
    max_depth:整型或none，可选，默认为无。树的最大深度。如果为none，则分割节点直到叶子数少于min_samples_split值。
    min_samples_spilt:整型，浮点型。可选，默认为2。
# -*-coding:utf-8 -*-

import numpy as np 
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

#创建一个随机数据集,该数据集由y=sinx和高斯噪声生成
rng = np.random.RandomState(1)
X = np.sort(5*rng.rand(80,1),axis=0)
y = np.sin(X).ravel()
y[::5] += 3*(0.5-rng.rand(16))

#创建决策树模型
regr = DecisionTreeRegressor(max_depth=5)
regr.fit(X,y)

#预测
X_test = np.arange(0.0,5.0,0.01)[:,np.newaxis] #生成0-5之间的测试数据
yHat = regr.predict(X_test)

#绘图
plt.figure()
plt.scatter(X,y,s=20,edgecolors="black",c="darkorange",label="data")
plt.plot(X_test,yHat,color="yellowgreen",label="max_depth=5",linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("sklearn")
plt.legend()
plt.show()
