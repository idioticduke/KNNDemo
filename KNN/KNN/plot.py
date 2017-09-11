import dataSet as ds
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt

datingDataMat,datingLabels = ds.file2matrix('F:\\计算机\\machinelearninginaction随书源代码\\Ch02\\datingTestSet2.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
plt.show()