import Service.LoadData as load
import Service.ChooseAttribute as chooseAttr
import numpy as np
import Utils.Model as trainModel
import Utils.seabornPlot as sns


def Main(readMethod, chooseFeatures=120, iterators=6, method=chooseAttr.filterFeatureScoreDecision, cweight=0.35):
    print('1 获取csv数据.......')
    X_train, y_train, all_dfindex, categeryAttribute = readMethod()
    index_need = np.array(all_dfindex)

    print('类别属性:', categeryAttribute)
    # 通过对每个属性的决策树评分来提高属性
    print('2 选择高效得属性.......')
    sortScores, sortfeatures, sortindexabs = chooseAttr.chooseFeture(X_train, y_train, index_need, categeryAttribute,
                                                                     method)

    needFeatures = sortfeatures[- chooseFeatures:]
    chooseindex = np.array(sortindexabs)[-chooseFeatures:]

    for i in range(len(chooseindex)):
            print(sortScores[- chooseFeatures:][i], '--', needFeatures[i])

    print('3 删除相关性属性....')


    removeAttributes = chooseAttr.removeFeature(X_train[:, chooseindex], y_train, selected_features=needFeatures)


    print(removeAttributes)

    bestIndex = []
    bestFeatures = []
    for m in range(len(needFeatures)):
        if (needFeatures[m] not in removeAttributes):
            bestIndex.append(chooseindex[m])
            bestFeatures.append(needFeatures[m])

    print(bestFeatures)

    print('4 训练模型，获取权重系数...')
    coefresult = []


    columns=[column for column in bestFeatures]
    categeryAttribute=[x for x in columns if x in categeryAttribute]
    coef, chooseindex = trainModel.trainProcess(X_train, y_train, chooseindex=bestIndex,chooseAttr=bestFeatures, categery=categeryAttribute,
                                                limit=0.50, cweight=cweight)
    coefresult.append(coef)
    meancoef = np.mean(coefresult, axis=0)

    print('5 测试..............')
    parmlit = list(range(45, 92, 5))
    parmResult = []
    print(bestFeatures)

    print(meancoef)
    for i in range(len(parmlit)):
        print('index:', parmlit[i] * 1.0 / 100.0)
        x, coef, chooseindex, rocvalue = trainModel.testProcess(X_train, y_train, bestIndex,chooseAttr=bestFeatures, limit=parmlit[i] / 100.0,
                                                                coeflist=meancoef, categery=categeryAttribute,
                                                                iterators=iterators, Cweight=cweight)
        result = np.array(x)[:, -3:-1]
        sum1 = 0;
        sum2 = 0;
        for j in range(len(result)):
            sum1 += float(result[j][0])  # 逾期率
            sum2 += float(result[j][1])  # 通过率

        parmResult.append([round(parmlit[i] * 1.0 / 100.0, 3), round(sum1 / iterators, 3),
                           round(sum2 / iterators, 3), round(rocvalue, 3)])
    return parmResult


parmResult = Main(load.loaddataAuto)

print('++++++++++++++result+++++++++++++++++')
print('阈值    逾期率   通过率  roc')
parmResult = np.array(parmResult)
print(parmResult)
sns.plotTwoDataNoLabel(list(parmResult[:, 0]), list(parmResult[:, 1]), list(parmResult[:, 2]), "loss", "pass",
                       "score limit", "ratio")
