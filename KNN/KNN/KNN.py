# -*- coding: utf-8 -*-

import dataSet as ds
import numpy as np
import operator

#group, labels = ds.createDataSet()

#datingDataMat,datingLabels = ds.file2matrix('F:\\计算机\\machinelearninginaction随书源代码\\Ch02\\datingTestSet2.txt')

#normDataSet, ranges, minVals = ds.autoNorm(datingDataMat)

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = ds.file2matrix('F:\\计算机\\machinelearninginaction随书源代码\\Ch02\\datingTestSet2.txt')
    normMat, ranges, minVals = ds.autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1))- dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #print (sortedClassCount)
    return sortedClassCount[0][0]



#datingClassTest()