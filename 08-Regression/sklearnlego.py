import numpy as np
import random
from sklearn import linear_model

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
函数说明:使用sklearn
parameters:
    无
return:
    无
"""   
def usesklearn():

    
    reg = linear_model.Ridge(alpha = .5)
    lgX,lgY = loadDataSet('lego.txt')
    reg.fit(lgX, lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2], reg.coef_[3]))    

if __name__ == '__main__':
    usesklearn()
