from numpy import *
import matplotlib
import matplotlib.pyplot as plt

"""
函数说明：加载数据集
parameters:
    fileName -文件名
return:
    dataMat -数据列表
"""
def loadDataSet(fileName):      
    dataMat = []                
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #将数据转换为float型数据
        dataMat.append(fltLine)
    return dataMat

"""
函数说明：计算向量欧氏距离
parameters:
    vecA -向量A
    vecB -向量B
return：
    欧氏距离
"""
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  #此处也可以使用其他距离计算公式

"""
函数说明：为给定数据集构建一个包含k个随机质心的集合
parameters:
    dataSet -数据集
    k -质心个数
return：
    centroids -质心列表
"""
def randCent(dataSet, k):
    n = shape(dataSet)[1] 
    centroids = mat(zeros((k,n))) #创建存储质心的矩阵，初始化为0
    for j in range(n):  #随机质心必须再整个数据集的边界之内
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ) #通过找到数据集每一维的最小和最大值
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1)) #生成0到1之间的随机数，确保质心落在边界之内
    return centroids

"""
函数说明：K-均值算法
parameters:
    dataSet -数据集
    k -簇个数
    distMeas -距离计算函数
    createCent -创建初始质心函数
return：
    centroids -质心列表
    clusterAssment -簇分配结果矩阵
"""
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]       #确定数据集中数据点的总数
    clusterAssment = mat(zeros((m,2)))  #创建矩阵来存储每个点的簇分配结果 
                                      #第一列记录簇索引值，第二列存储误差
    centroids = createCent(dataSet, k) #创建初始质心
    clusterChanged = True   #标志变量，若为True，则继续迭代
    while clusterChanged:
        clusterChanged = False 
        for i in range(m):      #遍历所有数据找到距离每个点最近的质心
            minDist = inf; minIndex = -1    
            for j in range(k):  #遍历所有质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])  #计算质心与数据点之间的距离
                if distJI < minDist:    
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2 #将数据点分配到距其最近的簇，并保存距离平方和
        print(centroids)    
        for cent in range(k):       #对每一个簇
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]   #得到该簇中所有点的值
            centroids[cent,:] = mean(ptsInClust, axis=0)    #计算所有点的均值并更新为质心
    return centroids, clusterAssment 

"""
函数说明：二分K-均值聚类算法
parameters:
    dataSet -数据集
    k -期望簇个数
    distMeas -距离计算函数
return：
    mat(centList) -质心列表矩阵
    clusterAssment -聚类结果
"""
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]  #得到数据集中样本点的个数
    clusterAssment = mat(zeros((m,2)))  #创建存储每个样本点的簇信息
    centroid0 = mean(dataSet, axis=0).tolist()[0] #最初将所有的数据看作一个簇，计算其均值
    centList =[centroid0]   #创建质心列表
    for j in range(m):  #遍历所有数据
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2  #计算每个样本点与质点的距离
    while (len(centList) < k):  #判断是否已经划分到用户指定的簇个数
        lowestSSE = inf     #将最小SSE设为无穷大
        for i in range(len(centList)):  #遍历所有簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#得到该簇所有数据的值
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) #在给定的簇上面进行K-均值聚类（k=2）
            sseSplit = sum(splitClustAss[:,1])      #计算被划分的数据的误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) #计算剩余数据的误差
            print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:        #如果该划分的误差平方和（SSE）值最小
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit  #将本次划分结果保存
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #由于使用二分均值聚类，会得到两个编号分别为0和1的结果簇
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit  #需要将这些簇编号更新为新加簇的编号
        print ('the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]   #更新质心列表
        centList.append(bestNewCents[1,:].tolist()[0])  #将新的质心添加至列表
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss    #更新新的簇分配结果
    return mat(centList), clusterAssment

"""
函数说明：地球表面两点之间的距离
parameters:
    vecA -向量A
    vecB -向量B
return:
    两个向量之间的球面距离
"""   
def distSLC(vecA, vecB):
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)   #使用球面余弦定理计算两点间的距离
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 

"""
函数说明：集成文本解析，聚类和画图
parameters:
    numClust -希望得到的簇个数
return:
    null
"""
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():         #类似于loadDataSet函数
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])     #文件第4列和第5列分别对应经度和维度
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)     #在数据集上进行二分均值聚类算法
    fig = plt.figure()      
    rect=[0.1,0.1,0.8,0.8]  #绘制矩形
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']  #构建标记形状的列表用于绘制散点图
    axprops = dict(xticks=[], yticks=[])    
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):   #遍历每个簇
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]   #使用索引来选择标记形状
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)    #使用"+"来标记质心
    plt.show()

def drawDataSet(dataMat,centList,myNewAssments):
    fig = plt.figure()      
    rect=[0.1,0.1,0.8,0.8]                                             #绘制矩形
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']  #构建标记形状的列表用于绘制散点图
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(2):                                                 #遍历每个簇
        ptsInCurrCluster = dataMat[nonzero(myNewAssments[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]          #使用索引来选择标记形状
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(centList[:,0].flatten().A[0], centList[:,1].flatten().A[0], marker='+', s=300)    #使用"+"来标记质心
    plt.show()
    
if __name__ =='__main__':
    #对地图上的点进行聚类
    #clusterClubs()
    #二分K-均值聚类算法
    dataMat = mat(loadDataSet('testSet.txt'))
    centList,myNewAssments =kMeans(dataMat,4)
    print(centList)
    fig = plt.figure()      
    rect=[0.1,0.1,0.8,0.8]  #绘制矩形
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']  #构建标记形状的列表用于绘制散点图
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(4):   #遍历每个簇
        ptsInCurrCluster = dataMat[nonzero(myNewAssments[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]   #使用索引来选择标记形状
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(centList[:,0].flatten().A[0], centList[:,1].flatten().A[0], marker='+', s=300)    #使用"+"来标记质心
    plt.show()
    
