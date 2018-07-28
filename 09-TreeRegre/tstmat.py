from numpy import *

"""
函数说明：读取文件内容存储到列表
parameters：
    fileName -文件路径
return：
    dataMat -数据矩阵
"""
def loadDataSet(fileName):      
    dataMat = []                
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')  #以tab键为分隔符分隔数据
        fltLine = list(map(float,curLine) ) #将每行的内容保存成一组浮点数
        dataMat.append(fltLine)
    return dataMat

"""
函数说明：给定特征值划分数据集
parameters:
    dataSet -数据集合
    feature -待切分的特征
    value -该特征的某个值
return:
    mat0 -划分后的数据集合
    mat1 -划分后的数据集合
"""
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]      #比给定特征值大的数据集合
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]     #比给定特征值小的数据集合
    return mat0,mat1

"""
函数说明：树构建函数
parameters:
    dataSet -数据集
    leafType -建立叶节点的函数
    errType -误差计算函数
    ops -包含树构建所需其他参数的元组
return:
    retTree -构建的树
    
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):   #确保dataSet为numpy矩阵，以对其进行数组过滤
    #****************主要实现函数***********************
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)        #找到最佳划分特征  
    
    if feat == None: return val                                         #如果划分时满足停止条件，feat会返回None值，val会是某类模型的值
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)                    #根据特征划分数据集合
    retTree['left'] = createTree(lSet, leafType, errType, ops)          #对新的数据集合分别继续递归调用createTree()函数
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree
"""    
if __name__ == '__main__':
    testMat = mat(eye(4))
    print("原始的数据集为:\n",testMat)
    mat0,mat1 = binSplitDataSet(testMat,1,0.5)
    print("划分后的数据集为:")
    print("mat0:\n",mat0)
    print("mat1:\n",mat1)
