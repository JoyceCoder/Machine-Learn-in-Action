#!/usr/bin/env python
#-*- coding:utf-8 -*-
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import random

"""
Desc：
    打开文件对其进行逐行解析
Parameters:
    fileName  文件名
Return:
    dataMat 数据矩阵
    labelMat 数据标签矩阵
"""
def loadDataSet(fileName):
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

"""
Desc:
    数据可视化
Parameters:
    dataMat 数据矩阵
    labelMat 数据标签矩阵
Return:
    NULL

def showDataSet(dataMat,labelMat):
    dataPlus = [];dataMinus = []    #定义正负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0: dataPlus.append(dataMat[i])
        else: dataMinus.append(dataMat[i])

    dataPlusNp = np.array(dataPlus)
    dataMinusNp = np.array(dataMinus)
    plt.scatter(np.transpose(dataPlusNp)[0],np.transpose(dataPlusNp)[1])
    plt.scatter(np.transpose(dataMinusNp)[0],np.transpose(dataMinus)[1])
    plt.show()

if __name__ == "__main__":
    dataMat, labelMat = loadDataSet('F:/python workspace/MachineLearning/06SVM/testSet.txt')
    showDataSet(dataMat, labelMat)
"""

"""
Desc:
    只要j不等于i，函数进行随机选择
Parameters:
    i 第一个alpha的下标
    m 总的alpha数目
Return:
    j 随机下标
"""
def selectJrand(i, m):
    j = i
    while(j==i):
        j = int(random.uniform(0,m))
    return j

"""
Desc:
    调整alpha的取值在某个范围之内
Parameters:
    aj 第j个alpha
    H alpha值的最大边界
    L alpha值得最小边界
Return:
    aj 调整后的alpha值
"""
def clipAlpha(aj, H, L):
    if aj >H:
        aj = H
    if aj < L:
        aj = L
    return aj

"""
Desc:
    简化版SMO
Parameters:
    dataMatIn 数据矩阵
    classLabels 数据标签矩阵
    C 常数
    toler 容错率
    maxIter 退出前最大的循环次数
Return:
    b  常数项
    alphas alpha矩阵
"""
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print ("L==H"); continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print ("eta>=0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print ("iteration number: %d" % iter)
    return b,alphas

"""
Desc:
    数据可视化
Parameters:
    dataMat 数据矩阵
    W 直线法向量
    b 直线截距
Return:
    Null
"""
def showClassifer(dataMat,w,b):
    dataPlus =[];dataMinus=[]
    for i in range(len(dataMat)):
        if labelMat[i] > 0: dataPlus.append(dataMat[i])
        else: dataMinus.append(dataMat[i])
    dataPlusNp=np.array(dataPlus)
    dataMinusNp=np.array(dataMinus)
    plt.scatter(np.transpose(dataPlusNp)[0], np.transpose(dataPlusNp)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(np.transpose(dataMinusNp)[0], np.transpose(dataMinusNp)[1], s=30, alpha=0.7) #负样本散点图

    x1=max(dataMat)[0]
    x2=min(dataMat)[0]
    a1,a2=w
    b=float(b)
    a1=float(a1[0])
    a2=float(a2[0])
    y1,y2=(-b-a1*x1)/a2,(-b-a1*x2)/a2
    plt.plot([x1,x2],[y1,y2])

    for i,alpha in enumerate(alphas):
        if alpha>0:
            x,y=dataMat[i]
            plt.scatter([x],[y],s=150,c='none',alpha=0.7,linewidth=1.5,edgecolor='red')
    plt.show()

def get_w(dataMat,labelMat,alphas):
    alphas,dataMat,labelMat = np.array(alphas),np.array(dataMat),np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1,-1).T,(1,2))*dataMat).T,alphas)
    return w.tolist()

if __name__ == '__main__':
    dataMat,labelMat=loadDataSet('F:/python workspace/MachineLearning/06SVM/testSet.txt')
    b,alphas=smoSimple(dataMat,labelMat,0.6,0.001,40)
    w = get_w(dataMat,labelMat,alphas)
    showClassifer(dataMat,w,b)

