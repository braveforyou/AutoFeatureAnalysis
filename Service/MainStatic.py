import Service.LoadData as load
import Service.ChooseAttribute as chooseAttr
import Utils.Model as trainModel
import numpy as np
import Utils.seabornPlot as sns


# chooseFeatures 选择多少属性  iterators测试数据 迭代轮数
def Main(readMethod, chooseFeatures=40, iterators=6, method=chooseAttr.filterFeatureScoreDecision, cweight=0.35):
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

    print('step5 testing..............')
    parmlit = list(range(45, 100, 5))
    parmResult = []

    for i in range(len(parmlit)):
        print('index:', parmlit[i] * 1.0 / 100.0)
        x, coef, chooseindex, rocvalue = trainModel.testProcess(bestIndex, X_train, y_train,
                                                                limit=parmlit[i] / 100.0, coeflist=meancoef,
                                                                iterators=iterators, Cweight=cweight)
        result = np.array(x)[:, -3:-1]
        sum1 = 0;
        sum2 = 0;
        for j in range(len(result)):
            sum1 += float(result[j][0])
            sum2 += float(result[j][1])

        parmResult.append([round(parmlit[i] * 1.0 / 100.0, 3), round(sum1 / iterators, 3),
                           round(sum2 / iterators, 3), round(rocvalue, 3)])
    return parmResult


parmResult = Main(load.loaddataSimple)

print('++++++++++++++result+++++++++++++++++')
parmResult = np.array(parmResult)
print(parmResult)
sns.plotTwoDataNoLabel(list(parmResult[:, 0]), list(parmResult[:, 1]), list(parmResult[:, 2]), "loss", "pass",
                       "score limit", "ratio")
