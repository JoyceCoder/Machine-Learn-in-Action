#!/usr/bin/env python
#-*- coding:utf-8 -*-

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
"""
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

if __nam__ == "__main__":
    dataMat, labelMat = loadDataSet('testSet.txt')
    showDataSet(dataMat, labelMat)
