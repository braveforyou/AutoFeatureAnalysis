from numpy import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import  DataFrame
sns.set(style="whitegrid")#python 可视化

# 获取模型的预测效果，包括精准率召回率等,输入预测值和实际值就好
def getPredictInfo(ypredict, yactual, column=0, weight=0, show=False, filename=''):
    countpos = 0;
    countneg = 0;

    for i in range(len(yactual)):
        if (yactual[i] == 0):
            countneg += 1
        else:
            countpos += 1

    posindex = [i for i, x in enumerate(yactual) if x == 1]
    negindex = [i for i, x in enumerate(yactual) if x == 0]
    ypredict = np.array(ypredict)
    tp = len([x for x in ypredict[posindex] if x == 1])
    fp = len([x for x in ypredict[posindex] if x == 0])
    tf = len([x for x in ypredict[negindex] if x == 0])
    ff = len([x for x in ypredict[negindex] if x == 1])

    text = '  逾期率:' + str(1 - tp / (tp + ff + 0.01)) + " 通过率:" + str((tp + ff) / (tp + fp + tf + ff))
    print(text)
    if show == True:
        pltbadDistribution(yactual, yactual, filename)

    return [column, weight, (tp + tf) / (tp + tf + ff + fp), tp / (tp + fp + 0.01), tp / (tp + ff + 0.01),
            tf / (tf + ff + 0.01), tf / (tf + fp + 0.01), 1 - tp / (tp + ff + 0.01), (tp + ff) / (tp + fp + tf + ff),
            text]


# 0,1之间分10段,画bad的占比趋势图
def pltbadDistribution(predict, actual, filename=""):
    result = np.zeros((13, 3))
    for i in range(len(predict)):
        index = int(predict[i] / 0.1)
        if (actual[i] == 0):
            result[index][0] += 1
        result[index][1] += 1
        result[index][2] = index
    result = np.array(result)
    selected_features = ['neg', 'total', 'range']
    crashes = DataFrame(result, columns=selected_features)

    f, ax = plt.subplots(figsize=(6, 15))
    sns.set_color_codes("pastel")
    sns.barplot(y="total", x="range", data=crashes,
                label="Total", color="b")

    sns.set_color_codes("muted")
    sns.barplot(y="neg", x="range", data=crashes,
                label="Alcohol-involved", color="b")

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 24), ylabel="",
           xlabel="Automobile collisions per billion miles")
    if (filename == ''):
        plt.show()
    else:
        plt.savefig(filename)


'''
# 描述DataFrame每一列得统计信息
二分类信息，如果缺失过多导致没有统计性，不采用为好。而多少数据才具有有效性与稳定性比较主观
或者说需要通过实验来逐步验证。首先如果说数据集中有效数据低于100条，那么统计性及稳定性是很差得。
所以，定义两个指标 1:绝对数值 2：相对占比

另外，对于这种缺失值太多的，应该怎么去衡量呢？作为评分，某些值得贡献要大才具有分辨性，但是模糊的部分也得要贡献足够小才行。
需要一个公式



'''

#查询要删除得字段
def descriptDataFrame(train, fliternull=0.8):
    all_index = [column for column in train]
    NullAttribute = []
    for i in range(len(all_index)):
        nonecount = 0
        records = train[all_index[i]]
        for j in range(len(records)):
            temprecord = records[j]
            if (pd.isnull(temprecord) or records[j] is None or records[j] == None or records[j] == NaN or records[
                j] == '' or records[j] == '-1' or records[j] == '-9' or records[j] == -1 or records[j] == -9):
                nonecount += 1
        print('index-',i,':', all_index[i])
        print('NoneRecords:', nonecount, "NoneRatio:", nonecount / len(train))
        if (nonecount / len(train) > fliternull and all_index[i].find('black') == -1):
            NullAttribute.append([all_index[i], nonecount / len(train), (len(train) - nonecount)])
        #print(train[all_index[i]].describe())

    NullAttribute = np.array(NullAttribute)
    print('=====NullAttribute ====')
    print(NullAttribute[:, 1].argsort())
    NullAttribute = NullAttribute[NullAttribute[:, 1].argsort()]
    print(NullAttribute)
    return list(NullAttribute[:, 0])




# ROC是对于连续值predStrengths变化而造成的TPR和FPR的变化曲线，这个predStrengths应该是算法中的一个关键的参数
# classLabels 是实际的label，而不是评分的label  predStrengths为list
def plotROC(predStrengths, classLabels, reverse=False, show=False):
    if reverse == True:
        classLabels = [(1 - x) for x in classLabels]
    predStrengths = array(predStrengths)
    cur = (1.0, 1.0)  # cursor
    ySum = 0.0  # variable to calculate AUC
    numPosClas = sum(array(classLabels) == 0)  # 正标签
    yStep = 1 / float(numPosClas);
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()  # get sorted index, it's reverse

    if show == True:
        fig = plt.figure()
        fig.clf()
        ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in list(sortedIndicies):
        if classLabels[index] == 0:
            delX = 0;
            delY = yStep;
        else:
            delX = xStep;
            delY = 0;
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        if show == True:
            ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)

    if show == True:
        ax.plot([0, 1], [0, 1], 'b--')
        plt.xlabel('False positive rate');
        plt.ylabel('True positive rate')
        plt.title('ROC curve for AdaBoost horse colic detection system')
        ax.axis([0, 1, 0, 1])
        plt.show()
    print("the Area Under the Curve is: ", ySum * xStep)
    return ySum * xStep
