import random
# 分布式梯度下降，允许所有进程在其数据batch上计算其模型的梯度，然后平均其梯度。
# 为了使更改几成熟使确保相似的收敛结果，首先对数据集进行分区
class Partitioning(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index
    def __len__(self):
        return len(self.index)
    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]
    

class DataPartitioner(object):
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed()
        data_len = len(data)
        indexs = [x for x in range(0, data_len)]
        rng.shuffle(indexs)
    
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexs[0:part_len])
            indexs = indexs[part_len:]
 
    def use(self, partition):
        return partition(self.data, self.partitions[partition])

