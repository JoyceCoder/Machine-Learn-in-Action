# -*-coding:utf-8 -*-

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def loadFile():
    return [line.split() for line in open('kosarak.dat').readlines()]

"""
函数说明：从列表到字典的类型转换函数
parameters：
    dataSet -数据集列表
return：
    retDict -数据集字典
"""
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1   #将列表项转换为forzenset类型并作为字典的键值，值为该项的出现次数
    return retDict

"""
类说明：FP树数据结构
function：
    __init__ -初始化节点
        nameValue -节点值
        numOccur -节点出现次数
        parentNode -父节点
    inc -对count变量增加给定值
    disp -将树以文本形式显示
"""
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue   #存放节点名字
        self.count = numOccur   #节点计数值
        self.nodeLink = None    #链接相似的元素值
        self.parent = parentNode    #当前节点的父节点
        self.children = {}  #空字典变量，存放节点的子节点
    
    def inc(self, numOccur):
        self.count += numOccur
    
    def disp(self, ind=1): #ind为节点的深度
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)   #递归遍历树的每个节点

"""
函数说明：FP树构建函数
parameters:
    dataSet -字典型数据集
    minSup -最小支持度
return：
    retTree -FP树
    headerTable -头指针表
"""
def createTree(dataSet, minSup = 1):
    headerTable = {}    #创建空字典，存放头指针
    for trans in dataSet: #遍历数据集
        for item in trans:  #遍历每个元素项
            headerTable[item] = headerTable.get(item,0)+dataSet[trans]  #以节点为key，节点的次数为值
    tmpHeaderTab = headerTable.copy()
    for k in tmpHeaderTab.keys():    #遍历头指针表
        if headerTable[k] < minSup:     #如果出现次数小于最小支持度
            del(headerTable[k])     #删掉该元素项
    freqItemSet = set(headerTable.keys())   #将字典的键值保存为频繁项集合
    if len(freqItemSet) == 0: return None, None #如果过滤后的频繁项为空，则直接返回
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] #使用nodeLink
        #print(headerTable)
    retTree = treeNode('Null Set',1,None) #创建树的根节点
    for tranSet, count in dataSet.items():    #再次遍历数据集
        localD = {} #创建空字典
        for item in tranSet:
            if item in freqItemSet: #该项是频繁项
                localD[item] = headerTable[item][0] #存储该项的出现次数，项为键值
        if len(localD) > 0: 
            orderedItems = [v[0] for v in sorted(localD.items(),key=lambda p: p[1],reverse = True)] #基于元素项的绝对出现频率进行排序
            #print(orderedItems)
            updateTree(orderedItems, retTree, headerTable, count)   #使用orderedItems更新树结构
    return retTree, headerTable

"""
函数说明：FP树生长函数
parameters:
    items -项集
    inTree -树节点
    headerTable -头指针表
    count -项集出现次数
return：
    None
"""
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:     #首先测试items的第一个元素项是否作为子节点存在
        inTree.children[items[0]].inc(count)  #如果存在，则更新该元素项的计数
    else:
        inTree.children[items[0]] = treeNode(items[0],count,inTree) #如果不存在，创建一个新的treeNode并将其作为子节点添加到树中
        if headerTable[items[0]][1] == None:    #将该项存到头指针表中的nodelink
            headerTable[items[0]][1] = inTree.children[items[0]]    #记录nodelink
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]]) #若已经存在nodelink，则更新至链表尾
    if len(items) > 1:
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)  #迭代，每次调用时会去掉列表中的第一个元素
    
"""
函数说明：确保节点链接指向树中该元素项的每一个实例
parameters:
    nodeToTest -需要更新的头指针节点
    targetNode -要指向的实例
return：
    None
"""
def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink!=None):   #从头指针表的nodelink开始，直到达到链表末尾
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode    #记录当前元素项的实例

"""
函数说明：上溯FP树
parameters：
    leafNode -节点
    prefixPath -该节点的前缀路径
return:
    None
"""
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None: #如果该节点的父节点存在
        prefixPath.append(leafNode.name)    #将其加入到前缀路径中
        ascendTree(leafNode.parent,prefixPath)  #迭代调用自身上溯

"""
函数说明：遍历某个元素项的nodelink链表
parameters：
    basePat -头指针表中元素
    treeNode -该元素项的nodelist链表节点
return:
    condPats -该元素项的条件模式基
"""
def findPrefixPath(basePat, treeNode):
    condPats = {} #创建空字典，存放条件模式基
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)    #寻找该路径下实例的前缀路径
        if len(prefixPath)>1:   #如果有前缀路径
            condPats[frozenset(prefixPath[1:])] = treeNode.count #记录该路径的出现次数，出现次数为该路径下起始元素项的计数值
            #此处需说明
        treeNode = treeNode.nodeLink
    return condPats

"""
函数说明：在FP树中寻找频繁项
parameters:
    inTree -FP树
    headerTable -当前元素前缀路径的头指针表
    minSup -最小支持度
    preFix -当前元素的前缀路径
    freqItemList -频繁项集
return:
    None
"""
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    #print("mineTreeHander:",headerTable)
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])] #按照出现次数从小到大排序
    #print("bigL:",bigL)
    for basePat in bigL:  #从最少次数的元素开始
        newFreqSet = preFix.copy()  #复制前缀路径
        newFreqSet.add(basePat)     #将当前元素加入路径
        #print ('finalFrequent Item: ',newFreqSet)    #append to set
        freqItemList.append(newFreqSet)     #将该项集加入频繁项集
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])    #找到当前元素的条件模式基
        #print ('condPattBases :',basePat, condPattBases)
        myCondTree, myHead = createTree(condPattBases, minSup)  #过滤低于阈值的item，基于条件模式基建立FP树
        #print ('head from conditional tree: ', myHead)
        if myHead != None: #如果FP树中存在元素项
            #print ('conditional tree for: ',newFreqSet)
            #myCondTree.disp(1)    
            # 递归的挖掘每个条件FP树，累加后缀频繁项集        
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)  #递归调用自身函数，直至FP树中没有元素
"""
FP树测试函数
"""        
def testFPtree():
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    print("字典数据集：\n",initSet)
    myFPtree, myHeaderTab = createTree(initSet,3)
    myFPtree.disp()
    return myHeaderTab

def testPrefix(myHeaderTab):
    print(findPrefixPath('x',myHeaderTab['x'][1]))
    print(findPrefixPath('z',myHeaderTab['z'][1]))
    print(findPrefixPath('r',myHeaderTab['r'][1]))

"""
找到频繁集测试函数
"""
def testAll():
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myTree, headerTab = createTree(initSet,3)
    freqItems = []
    mineTree(myTree,headerTab,3,set([]),freqItems)
    print("频繁项集为：\n",freqItems)

if __name__ == '__main__':
    #myHeaderTab = testFPtree()    
    #testPrefix(myHeaderTab)
    #testAll()
    parsedDat = loadFile()
    initSet = createInitSet(parsedDat)
    myFPtree,myHeaderTab = createTree(initSet,100000)
    myFreqList = []
    mineTree(myFPtree,myHeaderTab,100000,set([]),myFreqList)
    print("被10万人浏览过的报道数：\n",len(myFreqList))
    print("被10万人浏览过的报道：\n",myFreqList)