from numpy import *
import socket, struct
import threading

mutex = threading.Lock()

# 按照行读取
def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []
    fr = open(fileName, encoding='utf-8')
    for line in fr.readlines():
        line = line.replace('\n', '')
        dataMat.append(line)
    return dataMat


# 读取内容
def loadfileContent(fileName):
    fr = open(fileName, 'rb')
    f_read = fr.read()
    f_read_decod = f_read.decode('utf-8')
    return f_read_decod


# 输入二维array
def writearray(array1, path, columnindex=""):
    fl = open(path, 'a')
    fl.write("index," + str(columnindex) + "\n")
    for i in range(len(array1)):
        fl.write(str(columnindex[i]) + ",")
        for j in range(len(array1[0])):
            fl.write(str(array1[i, j]) + ",")
        fl.write("\n")
    fl.close()


def write(list1, path, isip=0, huanhang=1):
    fl = open(path, 'a')
    count = 0
    for i in list1:
        try:
            if (isip != 0):
                fl.write(str(i) + ";" + str(long2IP(i)))
            else:
                fl.write(str(i))
            if (huanhang == 1):
                fl.write("\n")
            if (i == len(list1) - 1):
                fl.write("\n")
        except:
            count += 1
    fl.close()




def writeListContent(list1, path, huanhang=0):
    mutex.acquire(10)  # 加锁
    fl = open(path, 'a')
    count = 0
    for content in list1:
        tempstring = ""
        if (isinstance(content, list)):
            for item in content:
                tempstring += str(item) + ","
        else:
            tempstring = content
        tempstring = tempstring[0:len(tempstring) - 1]
        try:
            fl.write(tempstring)
            if (huanhang == 1):
                fl.write("\n")
        except:
            count += 1
    fl.close()
    mutex.release()




# 将list转换为频率词典
def getdic(list1):
    dic1 = {}
    # 统计accounts出现的次数
    for i in range(len(list1)):
        if (dic1.get(list1[i]) == None):
            dic1[list1[i]] = 1
        else:
            dic1[list1[i]] += 1
    keys = tuple(dic1.keys())
    return dic1, keys


# 把list转换为partion连接的文本
def list2String(templist,partion=" "):
    content = ''
    for i in range(len(templist)):
        content += templist[i] + partion
    return content


# 获取停用词
def loadStopWords():
    try:
        stopwords = loadDataSet("D:\\stopwords.txt")
    except:
        stopwords = []
    stopwords.append('')
    stopwords.append('\\n')
    stopwords.append('\n')
    return stopwords


# 转换IP
def convertIp2int(ipstring):
    try:
        ipstring = ipstring.replace("\n", "")
        ips = ipstring.split('.');
        value = 0
        for i in [0, 1, 2, 3]:
            value = value + int(ips[3 - i]) * pow(256, i)
        return int(value)
    except Exception as e:
        return ipstring


# long 转换为IP
def long2IP(iplong):
    iplong = int(iplong)
    return socket.inet_ntoa(struct.pack('!L', iplong))


# 转换时间  00:00:00.112
def converttime(time):
    time2 = time.split('.');
    msc = time2[1]
    if len(msc) == 2:
        msc = float(msc) * 10
    else:
        if len(msc) == 1:
            msc = float(msc) * 100
        else:
            if len(msc) == 2:
                msc = float(msc)
    times = time2[0].split(':');
    value = 0
    for i in range(len(times)):
        value = value + float(times[i]) * pow(60, len(times) - i - 1)
    return int(value * 1000 + float(msc))


# 字典值排序，key  值
def sort_by_value(inputdic):
    items = inputdic.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort()
    return [backitems[i][1] for i in range(0, len(backitems))], [backitems[i][0] for i in range(0, len(backitems))]


# 随机抽取index,给定列表长度，通过partsize来抽取下表，分为训练集与测试集
def chooseIndex(rangesize, partsize=8):
    trainindex=[i for i in range(rangesize) if random.randint(1, partsize) != 3]
    testindex=[i for i in range(rangesize) if i not in trainindex]
    return trainindex, testindex
