#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

"""
使用y=x**2+2并加入一些随机误差生成的数据点
parameters:
    null
return:
    xMat -数据集
    yMat -标签集
"""
def loadDataSet():
    data = np.array([[ -2.95507616,  10.94533252],
       [ -0.44226119,   2.96705822],
       [ -2.13294087,   6.57336839],
       [  1.84990823,   5.44244467],
       [  0.35139795,   2.83533936],
       [ -1.77443098,   5.6800407 ],
       [ -1.8657203 ,   6.34470814],
       [  1.61526823,   4.77833358],
       [ -2.38043687,   8.51887713],
       [ -1.40513866,   4.18262786]])
    m = data.shape[0]  #样本个数
    xMat = data[:,0].reshape(-1,1)
    yMat = data[:,1].reshape(-1,1) #将array转换为矩阵
    return xMat,yMat

"""
对数据进行多项式处理
parameters:
    degree -多项式维度
    xMat -数据集
return：
    xPolyMat -处理后的数据
"""
def polyMatData(xMat,degree=2):
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    xPolyMat = poly_features.fit_transform(xMat)
    return xPolyMat

"""
使用线性回归模型
parameters:
    xMat -数据集
    yMat -标签集
return:
    null
"""
def linearRegre(xMat,yMat):
    rg = LinearRegression()
    rg.fit(xMat,yMat)
    return rg.intercept_,rg.coef_


if __name__=='__main__':
    xMat,yMat=loadDataSet()
    
    xPolyMat = polyMatData(xMat)
    print("多项式处理后的数据为：",xPolyMat)
    intercept,coef = linearRegre(xPolyMat,yMat)
    
    xPlot = np.linspace(-3,3,1000).reshape(-1,1)
    xPlotPoly = polyMatData(xPlot)
    yPlot = np.dot(xPlotPoly,coef.T)+intercept

    plt.plot(xPlot,yPlot,'r-')
    plt.plot(xMat,yMat,'b.')
    plt.show()
    
