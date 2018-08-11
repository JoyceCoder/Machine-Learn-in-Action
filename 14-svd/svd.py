from numpy import *
from numpy import linalg as la

def loadExData():
    return[[1,1,1,0,0],
           [2,2,2,0,0],
           [1,1,1,0,0],
           [5,5,5,0,0],
           [1,1,0,2,2],
           [0,0,0,3,3],
           [0,0,0,1,1]]
def loadExData1():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]    

def loadExData2():
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]
#(需说明列向量)
"""
函数说明：相似度计算函数(欧氏距离)
parameters：
    inA -列向量A
    inB -列向量B
return：
    两个向量的相似度
"""
def ecludSim(inA, inB):
    return 1.0/(1.0+la.norm(inA-inB))

"""
函数说明：相似度计算函数(皮尔逊相关系数)
parameters：
    inA -列向量A
    inB -列向量B
return：
    两个向量的相似度
"""
def pearsSim(inA, inB):
    if len(inA)<3: return 1.0   #是否存在三个或更多的点。两个向量是完全相关的，返回1
    #print(corrcoef(inA,inB,rowvar=0))
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]     #将数据归一化到0到1之间

"""
函数说明：相似度计算函数(余弦相似度)
parameters：
    inA -列向量A
    inB -列向量B
return：
    两个向量的相似度
"""
def cosSim(inA, inB):
    num = float(inA.T*inB)  #计算分子
    denom = la.norm(inA)*la.norm(inB)   #计算分母
    return 0.5+0.5*(num/denom)  #归一化

"""
函数说明：给定相似度计算方式，评估物品得分
parameters:
    dataMat -数据矩阵
    user -用户编号
    simMeas -相似度计算方法引用
    item -物品编号
return:
    用户对该物品的预估评分
"""
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]   #行为用户，列为物品。得到物品个数
    simTotal = 0.0;ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]    #记录用户对物品j的评分
        if userRating == 0: continue    #用户未对该物品评分，继续
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0] 
        #print(dataMat[:,item].A)
        #print(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))
        #print(nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0)))
        if len(overLap) == 0: similarity = 0    #两个物品没有重合元素，相似度为0
        else:
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])  #对重合元素进行相似度计算
        #print("物品%d和物品%d的相似度为%f" %(item,j,similarity))
        simTotal += similarity  #评分总和
        ratSimTotal +=similarity*userRating #相似度和当前用户评分的乘积
    if simTotal == 0 : return 0.0   
    else: return ratSimTotal/simTotal   #对数据归一，使得评分值在0-5之间


"""
函数说明：推荐函数
parameters:
    dataMat -数据矩阵
    user -用户编号
    N -产生推荐结果的个数
    simMeas -相似度计算方法
    estMethod -评估函数
return:
    返回N个推荐结果
"""
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod = standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]     #找到该用户还未评分的物品
    if len(unratedItems) == 0: return "你吃过的东西太多啦"
    itemScores = []
    for item in unratedItems:   
        estimatedScore = estMethod(dataMat,user,simMeas,item)   #计算该物品的预估得分
        itemScores.append((item,estimatedScore))    #存储到list
    return sorted(itemScores,key=lambda jj: jj[1],reverse=True)[:N] #将得分排序，返回前N个
    
"""
函数说明：基于SVD的评分估计
parameters:
    dataMat -数据矩阵
    user -用户编号
    simMeas -相似度计算方法
    item -物品编号
return:
    物品的估计评分
"""
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]   #获取物品个数
    simTotal = 0.0; ratSimTotal = 0.0
    u,sigma,vt = la.svd(dataMat)    #对数据矩阵奇异值分解
    sig4 = mat(eye(4)*sigma[:4])    #只利用包含了90%能量值的奇异值      
    #print(sigma)
    xformedItems = dataMat.T * u[:,:4] *sig4.I  #利用u矩阵将物品转换到低维空间
    for j in range(n):
        userRating = dataMat[user,j]    #得到用户对该物品的评分
        if userRating == 0 or j==item:continue   #如果评分为0或者物品与比较物品相同，则跳过
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)    #转换为列向量，并计算相似度
        #print("物品%d和物品%d的相似度为%f" %(item,j,similarity))
        simTotal += similarity  
        ratSimTotal += similarity*userRating  
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal   


"""
函数说明：打印矩阵
parameters:
    inMat -数据矩阵
    thresh -阈值界定深色和浅色
return:
    None
"""
def printMat(inMat, thresh=0.8):
    for i in range(32): #图片的像素为32*32
        for j in range(32):
            if float(inMat[i,j]) > thresh:  #如果大于阈值，则输出1
                print(1,end='') #python3,输出不换行
            else:
                print(0,end='')
        print('')

"""
函数说明：图像压缩
parameters:
    numSV -给定的奇异值数目
    thresh -阈值
return:
    None
"""
def imgCompress(numSV=3,thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i])) #从文件中以数值方式读入字符
        myl.append(newRow)
    myMat = mat(myl)
    print("**********初始矩阵**********")
    printMat(myMat, thresh)
    u,sigma,vt = la.svd(myMat)
    sigRecon = mat(zeros((numSV,numSV)))    #构建对角线上为sigma的numSV*numSV的矩阵
    for k in range(numSV):
        sigRecon[k,k]=sigma[k]
    reconMat = u[:,:numSV] * sigRecon *vt[:numSV,:] #重构矩阵
    print("**********使用%d个奇异值的重构矩阵**********" %numSV)
    printMat(reconMat,thresh)


"""
if __name__ == '__main__':
 
    #测试svd奇异值分解
    data=loadExData()
    u,sigma,vt=la.svd(data)
    print("奇异值矩阵：\n",sigma)
    sig3=mat([[sigma[0],0,0],[0,sigma[1],0],[0,0,sigma[2]]])
    print("重构矩阵:\n",u[:,:3]*sig3*vt[:3,:])

if __name__ == '__main__':
    #测试相似度计算
    myMat = mat(loadExData1())
    print("欧氏距离：")
    print(ecludSim(myMat[:,0],myMat[:,4]))
    print(ecludSim(myMat[:,0],myMat[:,0]))
    print('皮尔逊相关系数：')
    print(pearsSim(myMat[:,0],myMat[:,4]))
    print(pearsSim(myMat[:,0],myMat[:,0]))
    print('余弦相似度：')
    print(cosSim(myMat[:,0],myMat[:,4]))
    print(cosSim(myMat[:,0],myMat[:,0]))

if __name__ == '__main__':
    #测试推荐
    myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
    myMat[3,3]=2
    print(myMat)
    print(recommend(myMat,2))
    print(recommend(myMat,2,simMeas=ecludSim))
    print(recommend(myMat,2,simMeas=pearsSim))

if __name__ == '__main__':
    #测试svd推荐
    #myMat1 = mat(loadExData2())
    #print(recommend(myMat1,1,estMethod=svdEst))
    #print(recommend(myMat1,1,estMethod=svdEst,simMeas=pearsSim))
    #测试图像压缩
    imgCompress(2)

if __name__ == '__main__':
    u,sigma,vt = la.svd(mat(loadExData2()))
    print("奇异值矩阵为:\n",sigma)
    #计算总能量的90%
    sig2 = sigma**2
    parSig = sum(sig2)*0.9
    print("90%的能量为：\n",parSig)
    energy = 0.0
    while(energy < parSig):
        i = int(input("请输入所需奇异值个数："))
        energy = sum(sig2[:i])
        print("%d个元素所包含的能量为%f"%(i,energy))

if __name__ == '__main__':
    #测试svd推荐
    myMat1 = mat(loadExData2())
    print("使用svdEst,cosSim的推荐结果：\n",recommend(myMat1,1,estMethod=svdEst))
    print("使用svdEst,pearsSim的推荐结果：\n",recommend(myMat1,1,estMethod=svdEst,simMeas=pearsSim))
    print("使用standEst,pearsSim的推荐结果：\n",recommend(myMat1,1,simMeas=pearsSim))"""

if __name__ == '__main__':
    #测试推荐
    myMat = mat(loadExData1())
    myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
    myMat[3,3]=2
    print("修改后的矩阵：\n",myMat)
    print("余弦计算相似度推荐：\n",recommend(myMat,2))
    print("欧式相似度推荐：\n",recommend(myMat,2,simMeas=ecludSim))
    print("皮尔逊计算相似度推荐：\n",recommend(myMat,2,simMeas=pearsSim))