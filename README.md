# pytorch-
Pytorch分布式处理数据parallel
代码来自https://github.com/pytorch/pytorch/tree/master/torch/nn/parallel
使用pytorch训练网络模型，当使用多个GPU，数据和模型的分布式处理

1、DataParallel
    `torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)`
    DataParallel 自动分割数据，将作业订单发送到多个GPU上的多个模型。每GPU完成工作后DataParallel 将收集并合并结果。将最终结果返回。
    DataParallel将模型复制到所有的GPU,其中每个GPU消耗输入数据的不同分区。batch大小应大于所有可用GPU的数量。DataParallel使用多线程并行训练。没有使用torch.distributed中的线程通信API
2、DistributedDataParallel
    `CLASS torch.nn.parallel.DistributedDataParallel(module, device_ids=None, output_device=None, dim=0, broadcast_buffers=True, process_group=None, bucket_cap_mb=**25**, find_unused_parameters=False, check_reduction=False, gradient_as_bucket_view=False)`
    