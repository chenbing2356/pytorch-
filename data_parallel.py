"""
pytorch实现数据分布式训练源代码
"""
import operator
import torch
import warnings
from itertools import chain
from ..modules import Module
from .scatter_gather import scatter_kwargs, gather
from .replicate import replicate
from .parallel_apply import parallel_apply
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
    _get_devices_properties
)
from torch.nn.parallel import DistributedDataParallel as DDP

__all__ = ['DataParallel', 'data_parallel']

# GPU分配不均衡的问题
def _check_balance(device_ids):
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    device_ids = [_get_device_index(x, True) for x in device_ids]
    # _get_devices_properties返回一个包含设备属性信息的字典
    """
    {
    'name': 'GeForce GTX 1080 Ti',
    'capability_major': 6,
    'capability_minor': 1,
    'total_memory': 11264,
    'multi_processor_count': 28
    }
    """
    dev_props = _get_devices_properties(device_ids)    # 模型和数据分配到指定的每一个GPU上， 

    def warn_imbalance(get_prop):
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return


class DataParallel(Module):
    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__()
        torch._C._log_api_usage_once("torch.nn.parallel.DataParallel")
        # 判断返回"cuda"还是None
        device_type = _get_available_device_type()    
        if device_type is None:
            self.module = module    # 模型
            self.device_ids = []    # 指定的GPU
            return


        # 如果没有指定GPU，默认使用所有可用的GPU
        if device_ids is None:
            device_ids = _get_all_device_indices()
        
        # 如果没有指定输出设备的GPU，默认为指定设备的第一块
        if output_device is None:
            output_device = device_ids[0]
        

        self.dim = dim
        self.module = module     # 传入的模型
        self.device_ids = [_get_device_index(x, True) for x in device_ids]
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        _check_balance(self.device_ids)


        # 如果只指定了一个GPU，同时作为输出设备的GPU
        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)


    # 前向传播
    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DataParallel.forward"):

            # 没有可用的GPU，使用原来的模型来计算。
            if not self.device_ids:
                return self.module(*inputs, **kwargs)
            
            # 判断模型的参数和buffer都存在，前向传播中的参数和缓冲区
            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError("module must have its parameters and buffers "
                                       "on device {} (device_ids[0]) but found one of "
                                       "them on device: {}".format(self.src_device_obj, t.device))
                
            # 通过scatter函数将输入平均分配到每一个GPU上
            """
            tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]]) output
            tensor([[3, 1, 2, 0],
                    [1, 2, 0, 3]]) index_s
            tensor([[ 0.0093, -0.0758,  1.4261, -1.1239],
                    [-0.0637, -0.3633,  0.3970, -0.2359]]) input_s
            output = output.scatter(1, index_s, input_s)

            tensor([[-1.1239, -0.0758,  1.4261,  0.0093,  0.0000],
            [ 0.3970, -0.0637, -0.3633, -0.2359,  0.0000]]) output

            """
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # for forward function without any inputs, empty list and dict will be created
            # so the module can be executed on one device which is the first one in device_ids
            # 输入模型为空，返回一个空的list和dict,模型由第一个指定的GPU来执行
            if not inputs and not kwargs:
                inputs = ((),)
                kwargs = ({},)
            
            # 如果指定了一个GPU，就直接调用为并行的模型
            if len(self.device_ids) == 1:
                return self.module(*inputs[0], **kwargs[0])
            # 如果指定的GPU数量超过2
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])    # 将模型复制到多个GPU上
            outputs = self.parallel_apply(replicas, inputs, kwargs)    # 并行在多个GPU上计算模型
            return self.gather(outputs, self.output_device)    # 将数据聚合到一起，传送到output_device上

    # 复制模型，将副本放到每一个GPU设备上，传入模型和设备的ID列表
    def replicate(self, module, device_ids):
        return replicate(module, device_ids, not torch.is_grad_enabled())

    # 将输入数据和关键参数分散到每一个设备上，传入输入数据、关键字参数和设备ID列表
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    # 并行应用每个副本上的forward方法，传入副本列表、输入数据和关键字参数
    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    # 收集所有设备上的输出，并将它们合并成一个输出。传入设备输出列表和输出设备ID
    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.
    This is the functional version of the DataParallel module.
    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,) if inputs is not None else ()

    device_type = _get_available_device_type()

    if device_ids is None:
        device_ids = _get_all_device_indices()

    if output_device is None:
        output_device = device_ids[0]

    device_ids = [_get_device_index(x, True) for x in device_ids]
    output_device = _get_device_index(output_device, True)
    src_device_obj = torch.device(device_type, device_ids[0])

    for t in chain(module.parameters(), module.buffers()):
        if t.device != src_device_obj:
            raise RuntimeError("module must have its parameters and buffers "
                               "on device {} (device_ids[0]) but found one of "
                               "them on device: {}".format(src_device_obj, t.device))

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    # for module without any inputs, empty list and dict will be created
    # so the module can be executed on one device which is the first one in device_ids
    if not inputs and not module_kwargs:
        inputs = ((),)
        module_kwargs = ({},)

    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)