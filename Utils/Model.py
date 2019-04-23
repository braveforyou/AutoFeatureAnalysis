import numpy as np
from pandas import DataFrame
from sklearn.linear_model.logistic import LogisticRegression
import Utils.CommonUtil as comm
import Utils.CalculateTool as calt
import Utils.Evaluate as eval
import Utils.FormatData as format


# 规范数据范围
def filterData(predict):
    result = []
    for i in range(len(predict)):
        if (predict[i] > 1): predict[i] = 1
        if (predict[i] < -1): predict[i] = -1
        result.append(calt.getsigmoid(predict[i]))
    return result


# 对训练数据进行随机抽取
def randomPartSingleData(X_train, y_train, partsize=8):
    trainindex, testindex = comm.chooseIndex(len(X_train), partsize=partsize)
    X_trainNeed = X_train[trainindex]
    X_testNeed = X_train[testindex]
    y_trainNeed = y_train[trainindex]
    y_testNeed = y_train[testindex]
    return X_trainNeed, X_testNeed, y_trainNeed, y_testNeed


# 训练过程
def trainProcess(X_train, y_train, chooseindex=[],chooseAttr=[], categery=[],limit=0.5, cweight=0.25):
    AllWeights = []

    for i in range(3):
        # 随机划分数据集
        X_trainNeed, X_testNeed, y_trainNeed, y_testNeed = randomPartSingleData(X_train[:, chooseindex], y_train)
        X_trainNeed = DataFrame(X_trainNeed, columns=chooseAttr)
        X_testNeed = DataFrame(X_testNeed, columns=chooseAttr)
        # 对数据集进行归一化等操作
        X_trainNeed, X_testNeed, all_dfindex = format.convertTrainAndTestData(X_trainNeed, X_testNeed, 'scale',categery)
        # 进行逻辑回归
        lgr = LogisticRegression(C=cweight, solver='newton-cg')
        lgr.fit(X_trainNeed, y_trainNeed)
        AllWeights.append(lgr.coef_)

    # 训练N次，然后取系数的均值
    AllWeights = np.array(AllWeights)
    coef = np.mean(AllWeights, axis=0)
    lgr.coef_ = coef

    labelpredict = lgr.predict(X_testNeed)
    labelpredict = filterData(labelpredict)  # 转换到-1，1 范围
    labelpredict = np.array(labelpredict)
    mide = np.median(labelpredict)

    print('limit:', limit, 'mide:', mide, ' min:', np.min(labelpredict), ' max:', np.max(labelpredict))
    #labelpredict = format.adjust(labelpredict, limit)
    #eval.getPredictInfo(labelpredict, y_testNeed, show=True)
    return coef, chooseindex


# 测试当前获得变量得有效性 coeflist
def testProcess( X_train, y_train, chooseindex,chooseAttr,limit=0.4, coeflist=[],categery=[], iterators=10, Cweight=3.5):
    Xlist = []
    for i in range(iterators):
        X_trainNeed, X_testNeed, y_trainNeed, y_testNeed = randomPartSingleData(X_train[:, chooseindex], y_train,
                                                                                partsize=6)
        X_trainNeed = DataFrame(X_trainNeed, columns=chooseAttr)
        X_testNeed = DataFrame(X_testNeed, columns=chooseAttr)
        X_trainNeed, X_testNeed, all_dfindex = format.convertTrainAndTestData(X_trainNeed, X_testNeed, 'scale',categery)

        lgr = LogisticRegression(C=Cweight, solver='newton-cg')
        lgr.fit(X_trainNeed, y_trainNeed)
        lgr.coef_ = coeflist

        protoLabelPredict = calt.getLogisticValue(lgr.coef_, X_testNeed)
        rocvalue = eval.plotROC(protoLabelPredict, y_testNeed, True,False)
        #rocvalue = 0
        protoLabelPredict = np.array(protoLabelPredict).reshape(len(protoLabelPredict), 1)
        eval.pltbadDistribution(protoLabelPredict, y_testNeed, 'imgs/img' + str(i) + '.png')

        labelpredict = format.adjust(protoLabelPredict, limit)
        x = eval.getPredictInfo(labelpredict, y_testNeed, show=False)
        Xlist.append(x)
    return Xlist, [], chooseindex, rocvalue


