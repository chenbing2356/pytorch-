# 创建进程，初始化进程组
import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import network.ToyModel as ToyModel

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    # 初始化默认的分布式进程组，这还将初始化分布式程序包
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()

# 将DistributedDataParallel与模型并行组合，实现显式地将不同的模块放到不同的GPU上
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)

    # 将多GPU模型传递给DDP，不能设置device_ids和output_device.输入输出数据通过模型的forward()方法防止在适当的设备中

    def demo_model_parallel(rank, world_size):
        setup(rank, world_size)

        # setup mp_model and devices for this process
        dev0 = rank * 2
        dev1 = rank * 2 + 1
        mp_model = ToyMpModel(dev0, dev1)
        ddp_mp_model = DDP(mp_model)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        # outputs will be on dev1
        outputs = ddp_mp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(dev1)
        loss_fn(outputs, labels).backward()
        optimizer.step()

        cleanup()
    def run_demo(demo_fn, world_size):
        mp.spawn(demo_fn,
                args=(world_size,),
                nprocs=world_size,
                join=True)
        

    #保存和加载检查点
    def demo_checkpoint(rank, world_size):
        setup(rank, world_size)

        # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
        # rank 2 uses GPUs [4, 5, 6, 7].
        n = torch.cuda.device_count() // world_size
        device_ids = list(range(rank * n, (rank + 1) * n))

        model = ToyModel().to(device_ids[0])
        # output_device defaults to device_ids[0]
        ddp_model = DDP(model, device_ids=device_ids)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
        if rank == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 saves it.
        dist.barrier()
        # configure map_location properly
        rank0_devices = [x - rank * len(device_ids) for x in device_ids]
        device_pairs = zip(rank0_devices, device_ids)
        map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
        ddp_model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=map_location))

        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(device_ids[0])
        loss_fn = nn.MSELoss()
        loss_fn(outputs, labels).backward()
        optimizer.step()

        # Use a barrier() to make sure that all processes have finished reading the
        # checkpoint
        dist.barrier()

        if rank == 0:
            os.remove(CHECKPOINT_PATH)

    cleanup()
    if __name__ == "__main__":
        run_demo(demo_model_parallel, 4)
