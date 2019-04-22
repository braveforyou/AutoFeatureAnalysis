import numpy as np
import pandas as pd
import Utils.Consts as consts
import Utils.Evaluate as eval


# 过滤空值
def fillnull(df):
    allindex = [column for column in df]
    for i in range(len(allindex)):
        if isinstance(list(df[allindex[i]])[0], str): continue
        # 统计null得数量，如果不多，就选择填充中位数，否则填充-1
        try:
            countnull = 0
            for j in range(len(df)):
                if np.isnan(df[allindex[i]][j]):
                    countnull += 1
            # 小于5%得时候 用median
            if (countnull * 1.0 / len(allindex[i]) < 0.05):
                df[allindex[i]].fillna(df[allindex[i]].median(), inplace=True)
            else:
                df[allindex[i]].fillna(-1, inplace=True)
        except:
            # 异常情况，不处理这一组
            1
    selected_features = []
    for i in range(len(allindex)):
        if (allindex[i] != 'label' and allindex[i] != 'member_id' and allindex[i] != 'mobile'):
            selected_features.append(allindex[i])
    train = df[selected_features]
    return train


# 获取所有类别属性
def getCategeryFeature(df):
    selected_features = [column for column in df]
    categery = []
    count = 0

    for column in selected_features:  # 获取类别属性
        tempvalue = [1 for i in list(df[column]) if isinstance(i, str)]
        if (len(tempvalue) > 0):
            categery.append(column)
        count += 1
    return categery


import math


def loaddataAuto(filename='D:/hivefile.csv', filterMethod=fillnull):
    train = pd.read_csv(filename)
    # eval.descriptDataFrame(train)

    train = train[train["label"] != 2]  # label 0好 1坏 2拒绝
    y_train = train['label']
    all_dfindex = [column for column in train]
    y_train = np.array(y_train)
    selected_features = []
    print('filter attributes....')
    for i in range(len(all_dfindex)):
        if (all_dfindex[i] != 'label' and all_dfindex[i] != 'id' and all_dfindex[i] != 'member_id'
                and all_dfindex[i] != 'id_no_de' and all_dfindex[i] != 'mobile' and all_dfindex[
                    i] not in consts.filterAttribute):
            selected_features.append(all_dfindex[i])
    train = train[selected_features]

    train = filterMethod(train)  # 去除空值与不需要得属性


    train = np.array(train)

    width = len(train[0])
    height = len(train)
    for j in range(width):
        for i in range(height):  # 对于不同类型得列
            if (train[i][j] == '-9' or train[i][j] == '-99' or pd.isnull(train[i][j]) or
                    train[i][j] is None or train[i][j] == None
                    or train[i][j] == '-1' or train[i][j] == '-1.0' or train[i][j] == ''):
                train[i][j] = -1

    tempdatfarame=pd.DataFrame(train,columns=selected_features)
    categery = getCategeryFeature(tempdatfarame)  # 获取类别属性

    return train, y_train, selected_features, categery
