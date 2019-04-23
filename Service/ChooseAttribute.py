from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame
import numpy as np
import pandas as pd
import Utils.FormatData as format


def innerDatas(X_train, Y_train, attrIndex, selected_feature, categery=[]):
    dt = DecisionTreeClassifier(random_state=5, max_depth=40)
    temptrain = X_train[:, attrIndex]
    temptrain = temptrain.reshape(len(temptrain), 1)
    temptrainDataFrame = DataFrame(temptrain, columns=[selected_feature])
    if (selected_feature in categery):
        all_df = pd.get_dummies(temptrainDataFrame, columns=[selected_feature])
    else:
        all_df = temptrain

    score = cross_val_score(dt, all_df, Y_train, cv=3, scoring='accuracy')
    return np.mean(score)


# 树模型划分效果,通过树模型来对单列属性进行划分效果评估
def filterFeatureScoreDecision(X_train, Y_train, selected_features=[], categery=[]):
    scoreList = [innerDatas(X_train, Y_train, i, selected_features[i], categery) for i in range(len(selected_features))]

    sortindexabs = np.argsort(scoreList)
    sortScores = np.array(scoreList)[sortindexabs]
    sortFeatures = np.array(selected_features)[sortindexabs]
    return sortScores, sortFeatures, sortindexabs


# 判断属性重要程度并输出
def chooseFeture(X_train, y_train, all_dfindex, categery, chooseAttributeMethod):
    sortScores, sortfeatures, sortindexabs = chooseAttributeMethod(X_train, y_train, all_dfindex, categery)
    return sortScores, sortfeatures, sortindexabs


# 内部移除方法
def innerRemove(dt, X_train, Y_train, i, selected_features, alreadyfilter=[]):
    tempselect = []
    tempindex = []
    for j in range(len(selected_features)):
        if (j != i and (selected_features[j] not in alreadyfilter)):
            tempselect.append(selected_features[j])
            tempindex.append(j)

    temptrain = X_train[:, tempindex]
    emptrainDataFrame = DataFrame(temptrain, columns=[tempselect])
    tempxtrain, tempall_dfindex = format.convertTrainData(emptrainDataFrame)
    score = cross_val_score(dt, tempxtrain, Y_train, cv=5, scoring='accuracy')
    score = np.mean(score)

    return score


# 树模型划分效果,通过树模型来对单列属性进行划分效果评估
def removeFeature(X_train, Y_train, selected_features=[]):
    # 过滤过程中将无效属性放入其中，知道删除属性不造成准确度上升,这一步先删除了负向影响得属性

    alreadyfilter = ['baiduidquerytimesd30', 'lasttwoweekscallouttimes', 'tdidnumberbystage7dayscrossplatform',
                     'tdidnumberbystage30dayscrossplatform', 'called_call_count_anomaly_count_3',
                     'tdmobilecar30dayscrossplatform',
                     'sex', 'xinyanbehaviorloansorgcount', 'xinyanbehaviorloanscount', 'xinyanbehaviorlatestthreemonth',
                     'xinyanbehaviorlatestonemonthsuc', 'xinyanbehaviorhistorysucfee', 'shumei_credit_loan_overdues']
    # return alreadyfilter
    dt = DecisionTreeClassifier(random_state=5, max_depth=40)

    emptrainDataFrame = DataFrame(X_train, columns=[selected_features])

    currentColumn=[column for column in emptrainDataFrame]
    selected_features=[i for i in currentColumn if i not in alreadyfilter]
    emptrainDataFrame=emptrainDataFrame[selected_features]
    X_train=np.array(emptrainDataFrame)

    xtrain, all_dfindex = format.convertTrainData(emptrainDataFrame)
    scoreFull = cross_val_score(dt, xtrain, Y_train, cv=5, scoring='accuracy')
    scoreFull = np.mean(scoreFull)
    print('全部属性分值：', np.mean(scoreFull))

    scoreMax = scoreFull
    for i in range(len(selected_features)):
        print(i)
        scoreinner = innerRemove(dt, X_train, Y_train, i, selected_features, alreadyfilter)
        if (scoreinner >= scoreMax):
            scoreMax = scoreinner
            alreadyfilter.append(selected_features[i])
            print('findAttribute to filter:', selected_features[i], '  currentScore:', scoreMax)

    print('alreadyfilter:', alreadyfilter)
    return alreadyfilter
