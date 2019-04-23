from numpy import *
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing


# 融合训练集和预测集，统一标准化和onohot
def convertTrainAndTestData(traindata, testdata, pos='scale', categery=[]):
    if (categery == []):
        for column in traindata:  # 获取类别属性，对于类别属性后面需要进行数值化转换
            temp2 = [x for x in traindata[column] if str(x).find('-') != -1]
            temp = [x for x in traindata[column] if isinstance(x, str)]
            if (len(temp) > 0 or len(temp2) > 0):
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

    # 类别属性得数值化转换
    all_df = pd.get_dummies(xtrain, columns=categery)
    all_dfindex = [column for column in all_df]

    # 标准化方法   https://blog.csdn.net/csmqq/article/details/51461696
    if (pos == 'scale'):
        xtrain = preprocessing.scale(np.array(all_df))
    elif (pos == 'robust'):  # 为了处理那些离散得较大得值
        xtrain = preprocessing.robust_scale(np.array(all_df))
    else:
        xtrain = preprocessing.minmax_scale(np.array(all_df))
    return xtrain[:xlen], xtrain[xlen:], all_dfindex


# 融合训练集和预测集，统一标准化和onohot
def convertTrainData(traindata, pos='scale'):
    categery = []
    for column in traindata:  # 获取类别属性，对于类别属性后面需要进行数值化转换
        temp2 = [x for x in traindata[column] if str(x).find('-') != -1]
        typeList = [1 for x in traindata[column] if isinstance(x, str)]
        if (len(typeList) > 0 or len(temp2) > 0):
            categery.append(column)

    xtrain = np.array(traindata)
    xtrain = list(xtrain)

    all_dfindex = [column for column in traindata]
    xtrain = DataFrame(xtrain, columns=all_dfindex)

    # 类别属性得数值化转换
    all_df = pd.get_dummies(xtrain, columns=categery)
    all_dfindex = [column for column in all_df]

    # 标准化方法   https://blog.csdn.net/csmqq/article/details/51461696
    if (pos == 'scale'):
        xtrain = preprocessing.scale(np.array(all_df))
    elif (pos == 'robust'):  # 为了处理那些离散得较大得值
        xtrain = preprocessing.robust_scale(np.array(all_df))
    else:
        xtrain = preprocessing.minmax_scale(np.array(all_df))
    return xtrain, all_dfindex


# 通过阈值调整预测结果
def adjust(predict, limit):
    result = []
    for i in range(len(predict)):
        if (predict[i] >= limit):
            result.append(1)
        else:
            result.append(0)
    return result
