from functools import partial

import torch

class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self, store_input=False, store_output=False):
        self.store_input = store_input
        self.store_output = store_output

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch

# 在给定的模块上注册一个加载状态字典前的钩子（hook）。这个钩子会在模型加载状态字典之前被调用。
# 使用 partial 函数将 self.hook_fn 方法与 module 参数绑定，然后注册这个绑定后的函数作为钩子。
class PerChannelLoadHook:
    def __init__(self, module, hook_param=["scale", "zero_point"]):
        self.hook = module._register_load_state_dict_pre_hook(partial(self.hook_fn, module=module))
        self.hook_param = hook_param

    # hook_fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs, module)
    # 参数:
    # state_dict: 要加载的状态字典。
    # prefix: 状态字典中键（key）的前缀，用于定位特定模块的参数。
    # local_metadata: 与状态字典相关的元数据。
    # strict: 一个布尔值，指示加载状态字典时是否应该严格匹配键。
    # missing_keys, unexpected_keys, error_msgs: 分别用于记录缺失的键、意外的键和错误信息。
    # module: 触发钩子的模块。
    # 功能:
    # 检查模块的 ch_axis属性，如果为 - 1，表示没有按通道的参数，直接返回。
    # 遍历模块的参数和缓冲区，如果它们的名称在hook_param 列表中，则尝试在状态字典中找到对应的参数。
    # 如果找到了对应的参数，并且其形状与模块中当前参数的形状不匹配，则将当前参数重置为与状态字典中参数形状相同、数据类型和设备相同的全1张量。
    def hook_fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
                module):
        if module.ch_axis == -1:
            # no per-channel parameters
            return
        for module_key, param in module._parameters.items():
            if module_key not in self.hook_param:
                continue
            candidate = prefix + module_key
            if candidate in state_dict:
                input_param = state_dict[candidate]
                if param.shape != input_param.shape:
                    param.data = torch.ones_like(input_param, dtype=param.dtype, device=param.device)
        for module_key, param in module._buffers.items():
            if module_key not in self.hook_param:
                continue
            candidate = prefix + module_key
            if candidate in state_dict:
                input_param = state_dict[candidate]
                if param.shape != input_param.shape:
                    param.data = torch.ones_like(input_param, dtype=param.dtype, device=param.device)

    def close(self):
        self.hook.remove()
