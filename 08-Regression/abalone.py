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
计算最佳拟合直线
parameters:
    xArr -给定的输入值
    yArr -给定的输出值
return:
    ws -回归系数
"""
def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T    #将数据保存到矩阵中
    xTx = xMat.T*xMat                       #计算x.T *x
    if linalg.det(xTx) == 0.0:              #使用linalg.det()方法来判断它的行列式是否为零，即是否可逆
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)              #使用最小二乘法计算w值
    return ws

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
计算预测误差的平方和
parameters:
    yArr -给定y值
    yHatArr -预测y值
return:
    ((yArr-yHatArr)**2).sum() -误差矩阵
"""
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

if __name__=='__main__':
    abX,abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

    print("使用局部加权线性回归预测误差：")
    print("核为0.1时：",rssError(abY[0:99],yHat01.T))
    print("核为1时：",rssError(abY[0:99],yHat1.T))
    print("核为10时：",rssError(abY[0:99],yHat10.T))

    yHat01 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
    yHat1 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
    yHat10 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)

    print("使用局部加权线性回归预测误差在新数据上的表现：")
    print("核为0.1时：",rssError(abY[100:199],yHat01.T))
    print("核为1时：",rssError(abY[100:199],yHat1.T))
    print("核为10时：",rssError(abY[100:199],yHat10.T))

    ws = standRegres(abX[0:99],abY[0:99])
    yHat = mat(abX[100:199])*ws
    print("使用标准线性回归预测误差为：",rssError(abY[100:199],yHat.T.A))
