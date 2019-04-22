import Service.LoadData as load
import Utils.ChooseAttribute as chooseAttr
import Utils.Model as trainModel
import numpy as np
import Utils.CommonUtil as comm
import time


def writecoef(array1, path):
    fl = open(path, 'a')
    for i in range(len(array1)):
        fl.write(str(np.round(array1[i], 6)))
        fl.write("\n")
    fl.close()


# 按照行读取
def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []
    fr = open(fileName, encoding='utf-8')
    for line in fr.readlines():
        line = line.replace('\n', '')
        dataMat.append(line)
    return dataMat


def ProcessInit(readMethod, chooseFeatures=40, iterators=6, method=chooseAttr.filterFeatureScoreDecision, cweight=0.35):
    print('step1 loadData.......')
    X_train, y_train, all_dfindex, categeryAttribute = readMethod()
    index_need = np.array(all_dfindex)

    # 通过对每个属性的决策树评分来提高属性
    print('step2 chooseFeture.......')
    sortScores, sortfeatures, sortindexabs = chooseAttr.chooseFeture(X_train, y_train, index_need, categeryAttribute,
                                                                     method)
    print("sortScores:", sortScores[- chooseFeatures:])
    needFeatures = sortfeatures[- chooseFeatures:]
    chooseindex = np.array(sortindexabs)[-chooseFeatures:]
    print('step3 delete corrrelate Attr....(这步只算一次就可以)')

    removeAttributes = chooseAttr.removeFeature(X_train[:, chooseindex], y_train, selected_features=needFeatures)

    bestIndex = []
    bestFeatures = []
    for m in range(len(needFeatures)):
        if (needFeatures[m] not in removeAttributes):
            bestIndex.append(chooseindex[m])
            bestFeatures.append(needFeatures[m])

    print('step4 train and get coef...')
    coefresult = []
    coef, chooseindex = trainModel.trainProcess(X_train, y_train, chooseindex=bestIndex, limit=0.55, cweight=cweight)
    coefresult.append(coef)
    meancoef = np.mean(coefresult, axis=0)

    print(meancoef[0])
    # 写入参数和index
    comm.write(bestIndex, "bestindex.txt")

    writecoef(list(meancoef[0]), "bestcoef.txt")

    print('step5 testing..............')
    parmlit = list(range(45, 100, 5))
    parmResult = []

    for i in range(len(parmlit)):
        print('index:', parmlit[i] * 1.0 / 100.0)
        x, coef, chooseindex, rocvalue = trainModel.testProcess(bestIndex, X_train, y_train,
                                                                limit=parmlit[i] / 100.0, coeflist=meancoef,
                                                                iterators=iterators)
        result = np.array(x)[:, -3:-1]
        sum1 = 0;
        sum2 = 0;
        for j in range(len(result)):
            sum1 += float(result[j][0])
            sum2 += float(result[j][1])

        parmResult.append([round(parmlit[i] * 1.0 / 100.0, 3), round(sum1 / iterators, 3),
                           round(sum2 / iterators, 3), round(rocvalue, 3)])
    return parmResult


'''

parmResult = ProcessInit(load.loaddataSimple)

print('++++++++++++++result+++++++++++++++++')
parmResult = np.array(parmResult)
print(parmResult)
sns.plotTwoDataNoLabel(list(parmResult[:, 0]), list(parmResult[:, 1]), list(parmResult[:, 2]), "loss", "pass","score limit","ratio")

'''


def Predict(Xpredict, readMethod=load.loaddataSimple):
    begain = time.time()
    print('step1 loadData.......')
    X_train, y_train, all_dfindex, categeryAttribute = readMethod()

    Xpredict = [Xpredict]
    print(Xpredict)
    Xpredict = np.array(Xpredict)
    bestcoef = loadDataSet("./bestcoef.txt")
    bestIndex = loadDataSet("./bestindex.txt")
    bestIndex = [int(x) for x in bestIndex]
    bestcoef = [float(x) for x in bestcoef]

    predictlabel = trainModel.predictSingle(bestIndex, X_train, Xpredict, limit=0.8, coeflist=bestcoef)
    end = time.time()
    print('cost:', (end - begain))
    return predictlabel