from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame
import numpy as np
import pandas as pd


# 树模型划分效果,通过树模型来对单列属性进行划分效果评估
def filterFeatureScoreDecision(X_train, Y_train, selected_features=[], categery=[]):
    lgr = DecisionTreeClassifier(random_state=5, max_depth=100)
    scoreList = []

    for i in range(len(selected_features)):
        temptrain = X_train[:, i]
        temptrain = temptrain.reshape(len(temptrain), 1)
        temptrainDataFrame = DataFrame(temptrain, columns=[selected_features[i]])
        if (selected_features[i] in categery):
            all_df = pd.get_dummies(temptrainDataFrame, columns=[selected_features[i]])
        else:
            all_df = temptrain

        score = cross_val_score(lgr, all_df, Y_train, cv=5, scoring='accuracy')
        scoreList.append(np.mean(score))

    sortindexabs = np.argsort(scoreList)
    sortScores = np.array(scoreList)[sortindexabs]
    sortFeatures = np.array(selected_features)[sortindexabs]
    return sortScores, sortFeatures, sortindexabs


# 判断属性重要程度并输出
def chooseFeture(X_train, y_train, all_dfindex, categery, chooseAttributeMethod):
    sortScores, sortfeatures, sortindexabs = chooseAttributeMethod(X_train, y_train, all_dfindex, categery)
    return sortScores, sortfeatures, sortindexabs

import Utils.FormatData as format

# 树模型划分效果,通过树模型来对单列属性进行划分效果评估
def removeFeature(X_train, Y_train, selected_features=[]):
    # 过滤过程中将无效属性放入其中，知道删除属性不造成准确度上升,这一步先删除了负向影响得属性
    alreadyfilter = ['sqxhycp', 'called_call_people_anomaly_count_2']
    #alreadyfilter=[]
    return alreadyfilter
    lgr = DecisionTreeClassifier(random_state=5, max_depth=100)

    print(selected_features)
    emptrainDataFrame = DataFrame(X_train, columns=[selected_features])
    xtrain, all_dfindex=  format.convertTrainData(emptrainDataFrame)
    scoreFull = cross_val_score(lgr, xtrain, Y_train, cv=5, scoring='accuracy')
    scoreFull=np.mean(scoreFull)
    print('full attribute score:',np.mean(scoreFull))


    maxscore=0
    for i in range(len(selected_features)):
        tempselect=[]
        removeattrib=selected_features[i]
        tempindex=[]
        for j in range(len(selected_features)):
            if(j!=i and (selected_features[j] not in alreadyfilter) ):
                tempselect.append(selected_features[j])
                tempindex.append(j)

        temptrain = X_train[:,tempindex]
        emptrainDataFrame = DataFrame(temptrain, columns=[tempselect])
        tempxtrain, tempall_dfindex = format.convertTrainData(emptrainDataFrame)
        score = cross_val_score(lgr, tempxtrain, Y_train, cv=5, scoring='accuracy')
        score=np.mean(score)
        if(score>maxscore):maxscore=score
        if(score*1.0/maxscore>1):
            print(removeattrib,'remove score:', score)

    return alreadyfilter
