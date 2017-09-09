import KNN
import dataSet as ds
import numpy as np

group, labels = ds.createDataSet()
testSet = np.array([0.2, 0.2])

print(KNN.classify0(testSet, group, labels, 3))
