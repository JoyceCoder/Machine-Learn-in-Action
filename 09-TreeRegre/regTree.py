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
函数说明：生成叶节点
parameters:
    dataSet -数据集
return:
    均值
"""
def regLeaf(dataSet):
    return mean(dataSet[:,-1])  #在回归树中是计算目标变量的均值

"""
函数说明：误差计算函数
parameters:
    dataSet -数据集
return：
    总方差
"""
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]    #使用均方差乘以数据集中样本个数来计算总方差

"""
函数说明：树构建函数
parameters:
    dataSet -数据集
    leafType -建立叶节点的函数
    errType -误差计算函数
    ops -包含树构建所需其他参数的元组
return:
    retTree -构建的树
"""    
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
函数说明：找到最佳切分特征
parameters:
    dataSet -数据集
    leafType -建立叶节点的函数
    errType -误差计算函数
    ops -指定参数控制函数停止时机
return：
    bestIndex -切分特征
    bestValue -切分特征值
"""
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]  #tolS是容许的误差下降值，tolN是切分的最少样本数
    #如果所有的目标变量值相同，退出并返回该值
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #退出条件1，特征数目为1
        return None, leafType(dataSet)
    m,n = shape(dataSet)  
    S = errType(dataSet) #误差计算函数
    bestS = float('inf'); bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):    #对每个特征
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]): #对每个特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)  #返回指定特征和特征值的切分结果
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue #判断是否满足用户指定参数
            newS = errType(mat0) + errType(mat1)    #计算新的切分误差
            if newS < bestS:    #检查新误差能否降低误差
                bestIndex = featIndex  #更新最佳划分特征
                bestValue = splitVal    #更新最佳划分特征值
                bestS = newS    #更新最小误差
    if (S - bestS) < tolS:  #如果降低的误差差值小于tolS，即用户设定的阈值，则不切分直接创建叶节点
        return None, leafType(dataSet) #退出条件2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue) #得到最佳划分特征的切分的数据集合
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #退出条件3，检查两个切分后的子集大小是否小于用户定义的参数colN
        return None, leafType(dataSet)
    return bestIndex,bestValue #返回最佳划分特征及值

"""
函数说明：将数据格式化成目标变量Y和自变量X，并执行简单线性回归
parameters：
    dataSet -数据集
return：
    ws -线性回归系数矩阵
    X -数据集X值
    Y -目标值Y值
"""
def linearSolve(dataSet):
    m,n = shape(dataSet)    #得到数据集的样本个数和特征个数，默认最后一列为目标值
    X = mat(ones((m,n))); Y = mat(ones((m,1)))  #创建与数据集相同形状的全为1的矩阵，为添加x0=1
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]   #将dataSet的前n-1个特征值复制到X矩阵中，最后一列存到Y矩阵中
    xTx = X.T*X     #计算xTx
    if linalg.det(xTx) == 0.0:  #判断矩阵逆是否存在
        raise NameError('该矩阵无法转置，尝试修改下ops参数值')
    ws = xTx.I * (X.T * Y)          #计算权重系数矩阵
    return ws,X,Y

"""
函数说明：模型树中负责生成叶节点的模型
parameters:
    dataSet -数据集
return：
    ws -回归系数矩阵
"""
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

"""
函数说明：模型树中的误差计算函数
parameters：
    dataSet -数据集
return:
    yHat和y之间的平方误差
"""
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

"""
函数说明：测试输入变量是否是一棵树
parameters:
    obj -输入变量
return:
    布尔类型结果
"""
def isTree(obj):
    return (type(obj).__name__=='dict') #用于判断当前处理的节点是否为叶节点

"""
函数说明：塌陷处理（即返回树平均值）
parameters:
    tree -树结构
return:
    树平均值
"""
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])    #从上向下遍历树直到叶节点为止
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])   
    return (tree['left']+tree['right'])/2.0     #找到两个叶节点则返回它们的平均值
    
"""
函数说明：剪枝
parameters:
    tree -待剪枝的树
    testData -剪枝所需的测试数据
return：
    剪枝后的树结构
"""
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)    #首先需要确认测试集是否为空
    if (isTree(tree['right']) or isTree(tree['left'])): #如果有分支为子树，对测试数据进行划分
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)   #如果左分支为子树，则对该子树进行剪枝
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)   #如果右分支为子树，则对该子树进行剪枝
    #如果分支均为叶节点，我们可以进行合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])    #对测试数据进行划分
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +sum(power(rSet[:,-1] - tree['right'],2))    #计算未合并时的误差和
        treeMean = (tree['left']+tree['right'])/2.0     #计算当前叶节点的平均值
        errorMerge = sum(power(testData[:,-1] - treeMean,2))    #计算合并后的误差值
        if errorMerge < errorNoMerge:   #如果合并后的误差比不合并的误差小就进行合并操作
            print("merging")
            return treeMean
        else: return tree   #否则不合并
    else: return tree
"""
函数说明：对回归树叶节点预测
parameters:
    model -树
    inDat -为了与函数modelTreeEval保持一致性
return:
    float(model) -预测值
"""
def regTreeEval(model, inDat):
    return float(model) #由于回归树叶节点为常数值，将其转换为float型直接返回。

"""
函数说明：对模型树叶节点预测
parameters:
    model -树
    inData -测试数据集
return:
    float(X*model) -预测值
"""
def modelTreeEval(model, inDat):
    n = shape(inDat)[1] #得到特征个数
    X = mat(ones((1,n+1)))  #对输入数据进行格式化处理，在原数据矩阵上增加第0列
    X[:,1:n+1]=inDat    #将输入数据矩阵复制到新矩阵的1~n+1列
    return float(X*model)   #计算误差值

"""
函数说明：预测函数
parameters:
    tree -树结构
    inData -单个数据点
    modelEval -叶节点预测函数
return：
    预测值
"""
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)     #如果tree为叶节点，直接调用相应预测函数
    if inData[tree['spInd']] > tree['spVal']:   #根据特征值来划分数据
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)   #该节点为树继续遍历
        else: return modelEval(tree['left'], inData)    #若为叶节点，调用预测函数
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)

"""
函数说明：树构建函数
parameters:
    tree -树结构
    testData -测试数据集
    modelEval -对叶节点数据进行预测的函数引用
return:
    yHat -预测值矩阵
"""        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

if __name__ =='__main__':

    myData = loadDataSet('ex2.txt')
    myMat = mat(myData)
    regTree = createTree(myMat)
    print(regTree)

    #回归树剪枝
    myTree = createTree(myMat)
    myDatTest = loadDataSet('ex2test.txt')
    myMatTest = mat(myDatTest)
    newTree = prune(myTree,myMatTest)
    print(newTree)

    myData1 = loadDataSet('exp2.txt')
    myMat1 = mat(myData1)
    regTree1 = createTree(myMat1,modelLeaf,modelErr,(1,10))
    print(regTree1)
   

    #预测值
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat,ops=(1,20))
    yHat = createForeCast(myTree,testMat[:, 0])
    print("回归树相关系数:", corrcoef(yHat,testMat[:,1], rowvar=0)[0,1])

    myTree2 = createTree(trainMat,modelLeaf,modelErr,ops=(1,20))
    yHat = createForeCast(myTree2,testMat[:,0],modelTreeEval)
    print("模型树相关系数：",corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])

    ws,X,Y = linearSolve(trainMat)
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i,0]*ws[1,0]+ws[0,0]
    print("简单线性回归相关系数：",corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])

