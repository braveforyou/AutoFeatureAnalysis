import numpy as np
import pandas as pd
import Utils.Evaluate as eval
import Utils.Consts as consts
# 过滤空值
def fillnull(df):
    df['XinyanMaxOverdueDays'].fillna('other', inplace=True)  # 特殊处理
    allindex = [column for column in df]
    for i in range(len(allindex)):
        if isinstance(df[allindex[i]][0], str): continue
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
        if (allindex[i] != 'label' and allindex[i] != 'label2' and allindex[i] != 'shifou'
            and allindex[i] != 'QueryOrgCntD15' and allindex[i] != 'member_id' and allindex[i] != 'mobile'):
            selected_features.append(allindex[i])
    train = df[selected_features]
    return train


# 获取所有类别属性
def getCategeryFeature(df):
    selected_features = [column for column in df]
    categery = []
    for column in selected_features:  # 获取类别属性
        for j in range(len(df)):
            if (isinstance(df[column][j], str)):
                categery.append(column)
                continue
    return categery


def loaddataAuto(filterMethod=fillnull):
    train = pd.read_csv('D:\\needData.csv')

    y_train = train['label2']

    train = filterMethod(train)  # 去除空值与不需要得属性

    all_dfindex = [column for column in train]

    y_train = np.array(y_train)

    categery = getCategeryFeature(train)
    train = np.array(train)

    newList = []
    labelnew = []

    for i in range(len(y_train)):
        if (y_train[i] < 2):#过滤不合适得标签
            newList.append(list(train[i]))
            labelnew.append(y_train[i])

    labelnew = np.array(labelnew)
    newList = np.array(newList)


    for j in range(len(newList[0])):
        for i in range(len(newList)):
            if (newList[i][j] == '-1' or newList[i][j] == '-9' or newList[i][j] == '-99'):
                newList[i][j] = '-1'

    return newList, labelnew, all_dfindex, categery




def loaddataSimple():
    train = pd.read_csv('D:\\hivefile.csv')
    eval.descriptDataFrame(train)


    '''    
    train['XinyanLoansScore'].fillna(train['XinyanLoansScore'].median(), inplace=True)
    train['XinyanLoansOrgCount'].fillna(train['XinyanLoansOrgCount'].median(), inplace=True)
    train['XinyanLoanCount1Month'].fillna(train['XinyanLoanCount1Month'].median(), inplace=True)
    train['XinyanChargebackFail1Month'].fillna(train['XinyanChargebackFail1Month'].median(), inplace=True)
    train['XinyanConsfinMaxLimit'].fillna(train['XinyanConsfinMaxLimit'].median(), inplace=True)
    train['XinyanFinanceMaxLimit'].fillna(train['XinyanFinanceMaxLimit'].median(), inplace=True)
    train['XinyanConsfinCreditLimit'].fillna(train['XinyanConsfinCreditLimit'].median(), inplace=True)
    train['XinyanHistorySucFee'].fillna(train['XinyanHistorySucFee'].median(), inplace=True)
    train['XinyanLatestThreeMonth'].fillna(train['XinyanLatestThreeMonth'].median(), inplace=True)
    train['XinyanHistoryFailFee'].fillna(train['XinyanHistoryFailFee'].median(), inplace=True)
    train['XinyanLatestOneMonthSuc'].fillna(train['XinyanLatestOneMonthSuc'].median(), inplace=True)
    train['XinyanLoansOverdueCount'].fillna(train['XinyanLoansOverdueCount'].median(), inplace=True)
    train['ConsfinAvgLimit'].fillna(train['ConsfinAvgLimit'].median(), inplace=True)
    train['XinyanMaxOverdueDays'].fillna('other', inplace=True)
    train['BaiduLoan2ManyScore'].fillna(train['BaiduLoan2ManyScore'].median(), inplace=True)
    train['QueryOrgCntM6'].fillna(train['QueryOrgCntM6'].median(), inplace=True)
    train['HuaceAppStability7d'].fillna(train['HuaceAppStability7d'].median(), inplace=True)
    train['HuaceFinance7d'].fillna(train['HuaceFinance7d'].median(), inplace=True)
    train['HuaceLoan180d'].fillna(train['HuaceLoan180d'].median(), inplace=True)
    train['XinyanScore'].fillna(train['XinyanScore'].median(), inplace=True)
    train['XinyanConsfinOrgCount'].fillna(train['XinyanConsfinOrgCount'].median(), inplace=True)
    '''

    allindex = [column for column in train]
    selected_features = []

    y_train = train['label']
    y_train = np.array(y_train)




    for i in range(len(allindex)):
        if (allindex[i] != 'label' and allindex[i] != 'label2' and allindex[i] != 'shifou'
            and allindex[i] != 'QueryOrgCntD15' and allindex[i] != 'member_id' and allindex[i] != 'id_no_de'and allindex[i] != 'mobile' and consts in    consts.filterAttribute
    ):
            selected_features.append(allindex[i])
    train = train[selected_features]

    categery = []
    for column in selected_features:  # 获取类别属性
        if (isinstance(train[column][0], str)):
            categery.append(column)

    train = np.array(train)
    newList = []
    # 0 好  1坏
    labelnew = []
    for i in range(len(y_train)):
        if (y_train[i] < 2):
            newList.append(list(train[i]))
            labelnew.append(y_train[i])

    labelnew = np.array(labelnew)
    newList = np.array(newList)

    all_dfindex = selected_features

    for j in range(len(newList[0])):
        for i in range(len(newList)):
            if (newList[i][j] == '-1' or newList[i][j] == '-9' or newList[i][j] == '-99'):
                newList[i][j] = '-1'

    return newList, labelnew, all_dfindex, categery