
import numpy as np
import math


def getsigmoid(y):
    return 1.0 / (1 + math.exp(-y))


#通过已有的coef系数，来输出逻辑回归原始值
def getLogisticValue(coef,X):
    coef=np.mat(coef)
    X=np.mat(X)
    result=[]
    for i in range(len(X)):
        item=getsigmoid(coef*X[i].T)
        result.append(item)
    return result



# 获取均方差
def getSD(nlist):
    narray = np.array(nlist)
    sum1 = narray.sum()
    narray2 = narray * narray
    sum2 = narray2.sum()
    mean = sum1 / len(nlist)
    var = sum2 / len(nlist) - mean ** 2
    return np.sqrt(var)

