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
    numFeat = len(open(fileName).readline().split('\t')) - 1  
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
给定lambda下的岭回归求解，计算回归系数
parameters:
    xMat -给定x值
    yMat -给定y值
    lam -用户指定lambda，默认为0.2
return:
    ws -回归系数矩阵
"""
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat   #计算X.T * X
    denom = xTx + eye(shape(xMat)[1])*lam #计算xTx + I
    if linalg.det(denom) == 0.0:  #检查行列式是否为0
        print ("该矩阵为奇异矩阵，不可逆")
        return
    ws = denom.I * (xMat.T*yMat) #岭回归求解w
    return ws

"""
数据标准化过程
parameters:
    xArr -给定x值
    yArr -给定y值
return:
    wMat -所有的回归系数矩阵
"""
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T #数据存储到矩阵中
    yMean = mean(yMat,0)    #求y值的平均值
    yMat = yMat - yMean     
    #标准化处理
    xMeans = mean(xMat,0)   #所有的特征减去各自的均值并除以方差
    xVar = var(xMat,0)      
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30 #在30个不同的lambda下计算w回归系数
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

if __name__ =='__main__':
    abX,abY = loadDataSet('abalone.txt')
    rws = ridgeTest(abX,abY)
    fig = plt.figure() #绘图
    ax = fig.add_subplot(111)
    ax.plot(rws)
    plt.show()
