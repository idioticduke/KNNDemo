import KNN
import dataSet as ds
import numpy as np

'''
group, labels = ds.createDataSet()
testSet = np.array([0.2, 0.2])

print(KNN.classify0(testSet, group, labels, 3))
'''
def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = ds.file2matrix('F:\\计算机\\machinelearninginaction随书源代码\\Ch02\\datingTestSet2.txt')
    normMat, ranges, minVals = ds.autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = KNN.classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print ("You will probably like this person: ",resultList[classifierResult - 1])


classifyPerson()

