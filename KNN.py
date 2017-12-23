from numpy import *
import operator
import matplotlib.pyplot as plt

# 创建数据和label
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#实现KNN算法
#####################################################
def classify0(inX,dataSet,labels,k):
    #求出样本集的行数，也就是labels标签的数目
    dataSetSize = dataSet.shape[0]
    #构造输入值和样本集的差值矩阵
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    #计算欧式距离
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #求距离从小到大排序的序号
    sortedDistIndicies = distances.argsort()
    #对距离最小的k个点统计对应的样本标签
    classCount = {}
    for i in range(k):
        #取第i+1邻近的样本对应的类别标签
        voteIlabel = labels[sortedDistIndicies[i]]
        #以标签为key，标签出现的次数为value将统计到的标签及出现次数写进字典
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #对字典按value从大到小排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回排序后字典中最大value对应的key
    return sortedClassCount[0][0]
# group, labels = createDataSet()
# print(classify0([0,0], group, labels, 3))
# print(classify0([0.9, 0.8], group, labels, 3))
#####################################################





#将文本记录转换为numpy的解析程序
#####################################################
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  #得到文件行数
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
#使用matplotlib创建散点图
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])  #plot1
# ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels)) # 加上彩色标签的散点图
#lt.show()
# print(datingDataMat)
# print(datingLabels)
#####################################################




#归一化特征值（将数据的特征值归一化到0-1范围之间）
#####################################################
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))  # 特征值相除
    return normDataSet, ranges, minVals

normMat, ranges, minVals = autoNorm(datingDataMat)
#print(normMat)   # 输出归一化后的数据
#####################################################