import matplotlib.pyplot as plt
import numpy as  np
import pandas as pd
import seaborn as sns
import Utils.CommonUtil as comm


# 直方图
def countplot(datalist, labelx='x', labely='y'):
    sns.set_style("whitegrid")
    sns.countplot(datalist);
    plt.xlabel(labelx);
    plt.ylabel(labely);
    plt.show()


# 观察列表X的分布
def disSimple(x, filename="./1.png"):
    sns.set(color_codes=True)
    sns.distplot(x);
    if (filename == ''):
        plt.show()
    plt.savefig(filename)


# 比较同一种类型的两个分布
def compTwoDimension(x, y):
    sns.set(style="white", palette="muted", color_codes=True)
    sns.kdeplot(x)
    sns.kdeplot(x, color='r')
    sns.kdeplot(y, color='b')
    plt.legend();
    plt.show()


# 查看二维分布，可以看出是否有分辨性，适用于无标签
def plt2Dimension(data):
    df = pd.DataFrame(data, columns=["x", "y"])
    sns.jointplot(x="x", y="y", data=df);
    plt.show()


# 计算的时候非数值属性不做计算
def heatmap(df, show=True, filename=""):
    all_dfindex = [column for column in df]
    lenx = len(all_dfindex)
    dfData = df.corr()
    savedata = np.array(dfData)
    comm.writearray(savedata, "D:/heatmap.txt", all_dfindex)
    plt.subplots(figsize=(lenx, lenx))  #
    #  设置画面大小
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Reds")
    if (show):
        plt.show()
    else:
        if (filename == ""):
            plt.savefig('heatmap.png')
        else:
            plt.savefig(filename)


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t'))  # get number of fields
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat




# 二维散点图
def plotTwoDimenSion(XList, YList, labelx="X", labelY="Y"):
    fig = plt.figure()
    Xmat = np.array(XList)
    Ymat = np.array(YList)
    ax = fig.add_subplot(111)
    ax.scatter(Xmat, Ymat, marker='s', s=90)
    plt.title('decision stump test data')
    ax.set_xlabel(labelx)
    ax.set_ylabel(labelY)
    plt.show()
    return


# 二维散点图,Xmat是list，里面存的是ndarray的二维点集合
def plotMorePic(Xmat):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(Xmat)):
        tempx = np.array(Xmat[i])
        ax.scatter(tempx[:, 0], tempx[:, 1], marker='s', s=90)
    plt.title('decision stump test data')
    plt.show()
    return 0



# 比较两个分布，X相同 Y不同
def plotTwoDataNoLabel(x, y1, y2,label1="y1",label2="y2",xname="x",yname="y"):
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    plt.legend()
    plt.show()
    plt.xlabel(xname)
    plt.ylabel(yname)
    return



# 比较两个分布，X相同 Y不同
def plotTwoDataWitLabel(Xmat, Ymat, xmat2, ymat2, w, b):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Xmat, Ymat, marker='s', s=90)
    plt.title('decision stump test data')
    ax.scatter(xmat2, ymat2, marker='o', s=90)

    Xall = Xmat + xmat2  # 求点X轴的范围
    x = np.arange(min(Xall), max(Xall), 0.1)
    y = []
    for i in x:
        y.append((-w[0] * i - b) / w[1])
    ax.plot(x, y, color='red')
    plt.show()
    return 0


# 输入的是两个列表，label不同.打印出所有二维的图像是什么样的
def pltAllTwoDimension(X, X2, basePath=''):
    attributes = len(X[0])
    for i in range(attributes):
        for j in range(attributes):
            if (len(basePath) > 1):
                filename = basePath + "img" + str(i) + "-" + str(j) + ".png"
            else:
                filename = ""
            plotTwoData(X[:, i], X[:, j], X2[:, i], X2[:, j], filename)


# #画出所有得二维关系，只不过是通过增加了一个Z间隔而形成了三维得图像
def pltAll2Dimension(X, X2, basePath=''):
    attributes = len(X[0])
    Z1 = list(np.ones(len(X)).transpose())
    Z2 = list(np.zeros(len(X2)).transpose())

    for i in range(attributes):
        for j in range(attributes):
            if (len(basePath) > 1):
                filename = basePath + "img" + str(i) + "-" + str(j) + ".png"
            else:
                filename = ""
            print(filename)
            np.r_[X, Z1]  # 将X添加到Z=1得面
            np.r_[X2, Z2]  # 将X2添加到Z=2得面
            plot3dNew(X, X2, filename)

'''
1. 可以直观明了地识别数据中的异常值
2. 利用箱体图可以判断数据的偏态和尾重
3. 利用箱体图可以比较不同批次的数据形状
'''


#inputdf输入得dataframe，attrX x轴得属性名，attrY y轴得属性名  attrhue桶(可以试标签桶)，反正是需要进行对比得变量
#桶得图是对于attrhue 观察attrX，attrY
def pltBucket(inputdf,attrX,attrY,attrhue=''):
    sns.set(style="whitegrid", color_codes=True)
    np.random.seed(sum(map(ord, "categorical")))
    # palette 调色板 #分组绘制箱线图，分组因子是day，在x轴不同位置绘制 #分组箱线图，分子因子是time，
    # 不同的因子用不同颜色区分 # 相当于分组之后又分组
    if attrhue!='':
        sns.boxplot(y=attrY, x=attrX, hue=attrhue, data=inputdf)
    else:
        sns.boxplot(y=attrY, x=attrX, data=inputdf)
    plt.show()
#planets = sns.load_dataset("planets")
#pltBucket(planets,"distance","method")

# ++++++++++++++++++下方为内部调用，上方为外部接口+++++++++++++++++++++++++++++++



def plotTwoData(Xmat, Ymat, xmat2, ymat2, filepath=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(Xmat, Ymat, marker='s', s=90)
    plt.title('decision stump test data')
    ax.scatter(xmat2, ymat2, marker='o', s=90)
    if (len(filepath) > 1):
        plt.savefig(filepath)
    else:
        plt.show()
    plt.close()
    return 0


def plot3dNew(Data1, Data2, filename="", label1='label1', label2='label2'):
    Xmat = [float(x) for x in Data1[:, 0]];
    Ymat = [float(x) for x in Data1[:, 1]];
    zmat = [float(x) for x in Data1[:, 2]]
    Xmat2 = [float(x) for x in Data2[:, 0]];
    Ymat2 = [float(x) for x in Data2[:, 1]];
    zmat2 = [float(x) for x in Data2[:, 2]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xmat, Ymat, zmat, c='red', marker='o', label=label1);
    ax.scatter(Xmat2, Ymat2, zmat2, c='b', marker='^', label=label2)
    ax.set_xlabel('X Label');
    ax.set_ylabel('Y Label');
    ax.set_zlabel('Z Label')
    plt.legend()
    if (len(filename) > 1):
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()
