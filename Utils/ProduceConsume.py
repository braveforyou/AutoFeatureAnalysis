import queue
import random, threading, time

'''
#list1=[12,31,'22',213,'222']
#consume(list1,process=process)
作为通用的多线程处理方式

1.cntent指得是要消费得数据
2.method指得是消费得方法体是什么  消费方法需要自定义而后传入就可以
3.produce 可以替代content，形成公用得队列
'''

# 生产者类
class Producer(threading.Thread):
    def __init__(self, name, queue,threadnum):
        threading.Thread.__init__(self, name=name)
        self.data = queue
        self.threadnum=threadnum

    def run(self):
        for i in range(self.threadnum):
            print("%s is producing %d to the queue!" % (self.getName(), i))
            self.data.put(i)
            time.sleep(random.randrange(10) / 5)
        print("%s finished!" % self.getName())


#消费得执行体，对读入得参数进行处理
def process(arg):
    print(arg)


# 消费者类
count=[]
count.append(0)
def Consumer(id,q,method=process):
    while True:
        count[0] += 1
        if(count[0]%10==0):
            print(count[0])
        method(q.get())


#method 为处理得process，content为输入得数据
def consume(content,method,threads=16):
    q = queue.Queue(len(content))
    for i in range(len(content)):
        q.put(content[i])

    for j in range(threads):
        t = threading.Thread(target=Consumer, args=(j,q,method))
        t.start()

