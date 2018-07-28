import numpy as np
import random
"""
函数说明:打开一个用tab键分隔的文本文件
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
函数说明:数据标准化
Parameters:
    xMat - x数据集
    yMat - y数据集
Returns:
    inxMat - 标准化后的x数据集
    inyMat - 标准化后的y数据集
""" 
def regularize(xMat, yMat):
       
    inxMat = xMat.copy()                                                        #数据拷贝
    inyMat = yMat.copy()
    yMean = np.mean(yMat, 0)                                                    #行与行操作，求均值
    inyMat = yMat - yMean                                                        #数据减去均值
    inMeans = np.mean(inxMat, 0)                                                   #行与行操作，求均值
    inVar = np.var(inxMat, 0)                                                     #行与行操作，求方差
    # print(inxMat)
    print(inMeans)
    # print(inVar)
    inxMat = (inxMat - inMeans) / inVar                                            #数据减去均值除以方差实现标准化
    return inxMat, inyMat

"""
函数说明:计算平方误差
Parameters:
    yArr - 预测值
    yHatArr - 真实值
Returns:
"""
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

"""
函数说明:标准回归，计算回归系数w
Parameters:
    xArr - x数据集
    yArr - y数据集
Returns:
    ws - 回归系数
"""
def standRegres(xArr,yArr):

    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat                            #根据文中推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

"""
函数说明:岭回归
Parameters:
    xMat - x数据集
    yMat - y数据集
    lam - 缩减系数
Returns:
    ws - 回归系数
"""
def ridgeRegres(xMat, yMat, lam = 0.2):

    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

"""
函数说明:岭回归测试
Parameters:
    xMat - x数据集
    yMat - y数据集
Returns:
    wMat - 回归系数矩阵
"""
def ridgeTest(xArr, yArr):

    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    #数据标准化
    yMean = np.mean(yMat, axis = 0)                        #行与行操作，求均值
    yMat = yMat - yMean                                    #数据减去均值
    xMeans = np.mean(xMat, axis = 0)                    #行与行操作，求均值
    xVar = np.var(xMat, axis = 0)                        #行与行操作，求方差
    xMat = (xMat - xMeans) / xVar                        #数据减去均值除以方差实现标准化
    numTestPts = 30                                        #30个不同的lambda测试
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))    #初始回归系数矩阵
    for i in range(numTestPts):                            #改变lambda计算回归系数
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))    #lambda以e的指数变化，最初是一个非常小的数，
        wMat[i, :] = ws.T                                 #计算回归系数矩阵
    return wMat

"""
函数说明:交叉验证岭回归
Parameters:
    xArr - x数据集
    yArr - y数据集
    numVal - 交叉验证次数
Returns:
    wMat - 回归系数矩阵
"""
def crossValidation(xArr, yArr, numVal = 10):

    m = len(yArr)                                                                        #统计样本个数                       
    indexList = list(range(m))                                                            #生成索引值列表
    errorMat = np.zeros((numVal,30))                                                    #create error mat 30columns numVal rows
    for i in range(numVal):                                                                #交叉验证numVal次
        trainX = []; trainY = []                                                        #训练集
        testX = []; testY = []                                                            #测试集
        random.shuffle(indexList)                                                        #打乱次序
        for j in range(m):                                                                #划分数据集:90%训练集，10%测试集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)                                                #获得30个不同lambda下的岭回归系数
        for k in range(30):                                                                #遍历所有的岭回归系数
            matTestX = np.mat(testX); matTrainX = np.mat(trainX)                        #测试集
            meanTrain = np.mean(matTrainX,0)                                            #测试集均值
            varTrain = np.var(matTrainX,0)                                                #测试集方差
            matTestX = (matTestX - meanTrain) / varTrain                                 #测试集标准化
            yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)                        #根据ws预测y值
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))                            #统计误差
    meanErrors = np.mean(errorMat,0)                                                    #计算每次交叉验证的平均误差
    minMean = float(min(meanErrors))                                                    #找到最小误差
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]                                #找到最佳回归系数
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    meanX = np.mean(xMat,0); varX = np.var(xMat,0)
    unReg = bestWeights / varX                                                            #数据经过标准化，因此需要还原
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % ((-1 * np.sum(np.multiply(meanX,unReg)) + np.mean(yMat)), unReg[0,0], unReg[0,1], unReg[0,2], unReg[0,3]))




if __name__ == '__main__':
    lgX,lgY = loadDataSet('lego.txt')
    lgX1 = np.mat(np.ones((63,5)))              #添加对应常数项的特征X0(X0=1)，创建全1矩阵，lgX的形状为(63,4)
    lgX1[:,1:5]=np.mat(lgX)                     #将原数据矩阵lgX复制到新数据矩阵lgX1的1到5列
    
    ws = standRegres(lgX1,lgY)                  #在新数据集上进行标准化回归
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (ws[0],ws[1],ws[2],ws[3],ws[4])) #查看标准化回归模型(最小二乘法)
    crossValidation(lgX,lgY,10)                 #查看岭回归模型
    print(ridgeTest(lgX, lgY))                  #查看回归系数变化情况
