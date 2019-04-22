# AutoFeatureAnalysis
自动字段解析


包含以下内容

1.从hive中读取数据，最终转换为csv格式
2.对于每个属性通过树模型分析划分效果，排序
3.分析属性得相关性
4.随机数据，产生多个系数，取均值
5.组成实际模型


先通过LoadFromeHive到测试机器上拉去数据
kinit需要进行keroes授权

讲csv数据拉取到本地中

执行MainStatic主程序