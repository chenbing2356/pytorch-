import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import partition as part


from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network import Net

class MultiProcessing():
    def __init__(self, rank, size,backend, world_size):
        self.rank = rank
        self.size = size
        self.backend = backend
        self.world_size = world_size
    """
    创建两个进程
    分别加入一个进程
    分别运行run,再run中完成多进程通信
    """
    def run(rank, size):

        # 点对点通信，信息从一个进程被发送到另一个进程
        # 将tensor放到进程1，两个进程都从tensor(0)开始。然后进程0递增张量并将其发送到进程1，以便于它们都以tensor(1)结尾。进程1需要分配内存以存储它接收的数据。
        # send/recv被阻塞：两个进程都停止，直到通信完成。
        tensor = torch.zeros(1)
        if rank == 0:
            tensor += 1
            
        
        # 另外一种无阻塞的通信方式
        # 通过wait函数使自己再紫禁城执行过程中保持休眠状态。但是不知道合适将数据传递改其它进程，因此再req,wait()完成之前，既不应该修改发送的张量也 不应访问接受的张量以防止不确定的写入
        tensor = torch.zeros(1)
        req = None
        if rank == 0:
            tensor += 1
            # 将进程发送到进程1
            req = dist.isend(tensor=tensor, dst=1)
            print('rank 0 started sending')
        else:
            # 从进程0接收tensor
            req = dist.irecv(tensor=tensor, src=0)
            print('rank1开始接收')
        req.wait()
        print("rank", "Rank", "has data", "tensor[0]")


        # 进程组件通信,与点对点通信相反，集合允许跨组中所有进程的通信模式
        # 将0、1组成进程组，及那个各自进程中的tensor[1]相加，需要组中所有张量的总和，因此使用dist.reduce_op.SUM用作化简运算符
        group = dist.new_group([0, 1])
        tensor = torch.ones(1)
        # 将op操作应用于所有tensor，并将结果存储到所有进程中
        dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)    

        # 用distributed.scatter(tensor,scatter_list=None, astnc_op=False)复制第i个进程，实现分布式训练，例如将数据分成四份，并分别发送到不同的gpu上计算梯度。scatter函数可以用来将信息从src进程发送到其它进程上。
        # distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False) 从dt中的所有进程复制tensor。在分布式训练时，不同进程计算得到的梯度需要汇总到一个进程，并计算平均值以获得同意的梯度。
        # distributed.reduce(tensor,dst,op,group)将操作op（操作）应用于所有tensor,并将结果存储在dst中。
        # distributed.all_reduce(tensor, op, group)与reduce相同，但是结果存储在所有进程中
        # distributed.broadcast(tensor, src, group)将tensor从src复制到所有其它进程
        # distributed.all_gather(tensor_list, tensor, group)将所有进程中的tensor复制到tensor_list中
        


    def init_process(rank, size, fn, backend='gloo'):
        """初始化分布式环境"""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend, rank=rank, world_size=size)

    # if __name__ == "__main__":
    #     size = 2
    #     processes = []
    #     for rank in range(size):
    #         p = Process(target=init_process, arg=(rank, size, run))
    #         p.start()
    #         processes.append(p)
    # for p in processes:
    #     p.join()


if __name__ == '__main__':
    # 对数据进行分区
    def partition_dataset():
        dataset = datasets.MNIST('./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
        
        size = dist.get_world_size()
        bsz = 128 / float(size)
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = part(dataset, partition_sizes)
        partition = partition.use(dist.get_rank())
        train_set = torch.utils.data.DataLoader(partition,
                                            batch_size=bsz,
                                            shuffle=True)
        return train_set, bsz
    # 计算在所有rank上梯度的平均值
    def average_gradients(model):
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

    # partition_dataset()
    def run():
        torch.manual_seed(1234)
        train_set, bsz = partition_dataset()    # 划分数据集
        model = Net()    # 实例化network
        optimizer = optim.SGD(model.parameters(),    # 优化器
                            lr=0.01, momentum=0.5)
        from math import ceil
        num_batches = ceil(len(train_set.dataset) / float(bsz))    # 训练的batch_size
        for epoch in range(10):
            epoch_loss = 0.0
            for data, target in train_set:
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                epoch_loss += loss.item()
                loss.backward()
                average_gradients(model)
                optimizer.step()
            print('Rank ', dist.get_rank(), ', epoch ',
                epoch, ': ', epoch_loss / num_batches)
    run()

