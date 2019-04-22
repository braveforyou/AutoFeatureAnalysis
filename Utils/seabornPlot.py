import matplotlib.pyplot as plt


# 比较两个分布，X相同 Y不同
def plotTwoDataNoLabel(x, y1, y2,label1="y1",label2="y2",xname="x",yname="y"):
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    plt.legend()
    plt.show()
    plt.xlabel(xname)
    plt.ylabel(yname)
    return



