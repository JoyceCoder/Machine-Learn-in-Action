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
绘制最佳拟合直线
parameters:
    null
return:
    null
"""
def showLinerRegre():
    xArr ,yArr = loadDataSet('ex0.txt') #加载数据集
    ws = standRegres(xArr,yArr)         #得到回归系数
    
    xMat = mat(xArr);yMat = mat(yArr)   #将数据转换成numpy矩阵
    yHat = xMat * ws                    #计算y的预测值矩阵
    fig = plt.figure()    
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])   #flatten()返回一个折叠成一维的数组
    
    xCopy = xMat.copy()
    xCopy.sort(0)                       #如果直线上的数据点次数混乱，绘图会出现问题，故需要排序
    yHat = xCopy * ws
    ax.plot(xCopy[:,1],yHat,c='red')
    plt.title('Dataset')
    plt.xlabel('x')
    plt.show()
    
if __name__ =='__main__':
    showLinerRegre()
