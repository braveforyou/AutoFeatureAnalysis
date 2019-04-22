from numpy import *
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import preprocessing



# 融合训练集和预测集，统一标准化和onohot
def convertTrainAndTestData(traindata, testdata,pos='scale'):
    categery = []
    for column in traindata:#获取类别属性，对于类别属性后面需要进行数值化转换
        if (isinstance(traindata[column][0], str)):
            categery.append(column)

    xtrain = np.array(traindata)
    xtrain = list(xtrain)
    testdata = np.array(testdata)
    xlen = len(xtrain)

    for i in range(len(testdata)):
        xtrain.append(testdata[i])
    xtrain = np.array(xtrain)

    all_dfindex = [column for column in traindata]
    xtrain = DataFrame(xtrain, columns=all_dfindex)

    #类别属性得数值化转换
    all_df = pd.get_dummies(xtrain, columns=categery)
    all_dfindex = [column for column in all_df]

    #标准化方法   https://blog.csdn.net/csmqq/article/details/51461696
    if(pos=='scale'):
        xtrain = preprocessing.scale(np.array(all_df))
    elif(pos=='robust'):#为了处理那些离散得较大得值
        xtrain=preprocessing.robust_scale(np.array(all_df))
    else:
        xtrain = preprocessing.minmax_scale(np.array(all_df))
    #xtrain=xtrain+pos#添加偏移，主要用于使所有值变成正值
    return xtrain[:xlen], xtrain[xlen:],all_dfindex


# 融合训练集和预测集，统一标准化和onohot
def convertTrainData(traindata,pos='scale'):
    categery = []
    for column in traindata:#获取类别属性，对于类别属性后面需要进行数值化转换
        if (isinstance(traindata[column][0], str)):
            categery.append(column)

    xtrain = np.array(traindata)
    xtrain = list(xtrain)
    xlen = len(xtrain)

    all_dfindex = [column for column in traindata]
    xtrain = DataFrame(xtrain, columns=all_dfindex)

    #类别属性得数值化转换
    all_df = pd.get_dummies(xtrain, columns=categery)
    all_dfindex = [column for column in all_df]

    #标准化方法   https://blog.csdn.net/csmqq/article/details/51461696
    if(pos=='scale'):
        xtrain = preprocessing.scale(np.array(all_df))
    elif(pos=='robust'):#为了处理那些离散得较大得值
        xtrain=preprocessing.robust_scale(np.array(all_df))
    else:
        xtrain = preprocessing.minmax_scale(np.array(all_df))
    #xtrain=xtrain+pos#添加偏移，主要用于使所有值变成正值
    return xtrain,all_dfindex





# 通过阈值调整预测结果
def adjust(predict, limit):
    result = []
    for i in range(len(predict)):
        if (predict[i] >= limit):
            result.append(1)
        else:
            result.append(0)
    return result