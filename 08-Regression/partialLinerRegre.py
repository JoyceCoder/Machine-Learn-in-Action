#!/usr/bin/env python
#-*- coding:utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt

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

"""
计算回归系数
parameters:
    testPoint -待预测数据
    xArr -给定输入值
    yArr -给定输出值
    k -高斯核的k值，决定对附近的点赋予多大的权重
return:
    testPoint * ws -回归系数的估计值
"""
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T #读入数据到矩阵
    m = shape(xMat)[0]
    weights = mat(eye((m))) #创建对角权重矩阵，该矩阵为方针，阶数为样本点个数
    for j in range(m):  #遍历整个数据集
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T/(-2.0*k**2)) #计算每个样本点对应的权重值，随着样本点与待预测点距离的递增，权重将以指数级衰减
    xTx = xMat.T*(weights*xMat)
    if linalg.det(xTx) ==0.0: #判断矩阵是否可逆
        print("该矩阵不可逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint*ws

"""
测试函数
parameters:
    testArr -测试数据集
    xArr -给定输入值
    yArr -给定输出值
    k -高斯核的k值
return:
    yHat -预测值
"""
def lwlrTest(testArr, xArr, yArr,k=1.0):
    m = shape(xArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

"""
绘制多条局部加权回归曲线
parameters:
        无
returns:
        无
 """
def plotlwlrRegression():
    xArr, yArr = loadDataSet('ex0.txt')                                    #加载数据集
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)                            #根据局部加权线性回归计算yHat
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)                            #根据局部加权线性回归计算yHat
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)                            #根据局部加权线性回归计算yHat
    xMat = mat(xArr)                                                    #创建xMat矩阵
    yMat = mat(yArr)                                                    #创建yMat矩阵
    srtInd = xMat[:, 1].argsort(0)                                        #排序，返回索引值
    xSort = xMat[srtInd][:,0,:]
    fig, axs = plt.subplots(nrows=3, ncols=1,sharex=False, sharey=False, figsize=(10,8))                                        
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c = 'red')                        #绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c = 'red')                        #绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c = 'red')                        #绘制回归曲线
    axs[0].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'green', alpha = .5)                #绘制样本点
    axs[1].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'green', alpha = .5)                #绘制样本点
    axs[2].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'green', alpha = .5)                #绘制样本点
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title(u'lwlr,k=1.0')
    axs1_title_text = axs[1].set_title(u'lwlr,k=0.01')
    axs2_title_text = axs[2].set_title(u'lwlr,k=0.003')
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')  
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')  
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')  
    plt.xlabel('X')
    plt.show()

if __name__ == '__main__':
    plotlwlrRegression()
