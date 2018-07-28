#!/usr/bin/env python
#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from numpy import *
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

"""
打开一个用tab键分隔的文本文件
parameters:
    fileName -文件名
return:
    dataMat -数据矩阵
    labelMat -目标值向量
"""
def loadDataSet(fileName):      
    numFeat = len(open(fileName).readline().split('\t')) - 1 #得到列数，不包括最后一列，默认最后一列值为目标值
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

if __name__ =='__main__':
    dataX, dataY =loadDataSet('ex0.txt')
    matX=mat(dataX);matY=mat(dataY).T #将数据保存到矩阵中
    regr = linear_model.LinearRegression()  #生成线性回归模型
    regr.fit(matX,matY)  #填充训练数据 matX(n_samples,n_features);matY(n_samples,n_targets)

    xCopy = matX.copy()
    xCopy.sort(0)
    predictY = regr.predict(xCopy) #得到模型预测值

    plt.scatter(matX[:,1].flatten().A[0],matY[:,0].flatten().A[0],s=20,color='green',alpha=.5) #绘制散点图
    plt.plot(xCopy[:,1],predictY,color='red',linewidth=1) #绘制最佳拟合直线
 
    plt.xticks(())
    plt.yticks(())

    plt.show()

    
    
