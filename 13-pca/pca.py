from numpy import *
import matplotlib.pyplot as plt
"""
函数说明：加载数据集
parameters:
    fileName -文件名
    delim -分隔符
return:
    mat(datArr) -数据矩阵
"""
def loadDataSet(fileName, delim = '\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]      #对读入数据以\t分隔存储到列表中
    datArr = [list(map(float,line)) for line in stringArr]   #使用两个list来构建矩阵
    return mat(datArr)

"""
函数说明：PCA算法实现
parameters:
    dataMat -用于进行PCA操作的数据集
    topNfeat -应用的N个特征
return:
    lowDataMat -将维后的数据集
    reconMat -重构的数据集（用于调试）
"""
def pca(dataMat, topNfeat = 9999999):
    meanVals = mean(dataMat, axis = 0)  #计算数据平均值
    meanRemoved = dataMat - meanVals    #去平均值
    covMat = cov(meanRemoved, rowvar = 0 ) #计算协方差
    eigVals, eigVects = linalg.eig(mat(covMat)) #计算协方差矩阵的特征值和特征向量
    eigValInd = argsort(eigVals) #对特征值从小到大排序，并提取对应的index
    eigValInd = eigValInd[:-(topNfeat+1):-1]    #对特征排序结果逆序
    redEigVects = eigVects[:,eigValInd]     #根据特征值排序结果得到topNfeat个最大的特征向量
    lowDataMat = meanRemoved * redEigVects  #数据降维
    reconMat = (lowDataMat * redEigVects.T) + meanVals      #数据重构
    return lowDataMat, reconMat

"""
函数说明：绘制数据集
parameters:
    dataMat -原始数据集
    reconMat -重构数据集
return:
    None
"""
def drawDataSet(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^',s=90)
    ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o',s=50,c='red')
    plt.show()

"""
函数说明：替换Nan为平均值
parameters:
    None
return:
    datMat -补充缺失值后的数据矩阵  
"""
def replaceNanWithMean():
    datMat = loadDataSet('secom.data',' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])     #计算每个特征项的均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal    #将空值替换为均值
    return datMat

if __name__ =='__main__':
    #dataMat = loadDataSet('testSet.txt')
    dataMat = replaceNanWithMean()
    #pca部分代码
    meanVals = mean(dataMat, axis = 0)  #计算数据平均值
    meanRemoved = dataMat - meanVals    #去平均值
    covMat = cov(meanRemoved, rowvar = 0 ) #计算协方差
    eigVals, eigVects = linalg.eig(mat(covMat))
    print("协方差矩阵的特征值结果：\n",eigVals)
    #总步骤
    lowDMat, reconMat = pca(dataMat,20)
    print("降维后的矩阵形状：\n",shape(lowDMat))
    drawDataSet(dataMat, reconMat)
    
                          


