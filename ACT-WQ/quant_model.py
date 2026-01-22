import torch
import torch.nn as nn
from mqbench.quant_modules import QuantConv2d, QuantLinear
from mqbench.gptq_quant_modules import GPTQConv2d, GPTQLinear
from typing import Dict
from mqbench.utils.scheme import QuantizeScheme
from mqbench.observer import *
from mqbench.fake_quantize import *
from mqbench.utils.utils import ModuleHook
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

ObserverDict = {
    'MinMaxObserver':           MinMaxObserver,                                    # noqa: E241
    'EMAMinMaxObserver':        EMAMinMaxObserver,        # More general choice.   # noqa: E241
    'MinMaxFloorObserver':      MinMaxFloorObserver,      # For Vitis HW           # noqa: E241
    'PoTModeObserver':          PoTModeObserver,   # For Vitis HW           # noqa: E241
    'EMAQuantileObserver':      EMAQuantileObserver,      # Quantile observer.     # noqa: E241
    'ClipStdObserver':          ClipStdObserver,          # Usually used for DSQ.  # noqa: E241
    'LSQObserver':              LSQObserver,              # Usually used for LSQ.  # noqa: E241
    'MSEObserver':              MSEObserver,                                       # noqa: E241
    'EMAMSEObserver':           EMAMSEObserver,                                    # noqa: E241
}

FakeQuantizeDict = {
    'FixedFakeQuantize': FixedFakeQuantize,      # Unlearnable scale/zeropoint  # noqa: E241
    'LearnableFakeQuantize': LearnableFakeQuantize,  # Learnable scale/zeropoint    # noqa: E241
    'NNIEFakeQuantize':      NNIEFakeQuantize,       # Quantize function for NNIE   # noqa: E241
    'DoReFaFakeQuantize':    DoReFaFakeQuantize,     # Dorefa                       # noqa: E241
    'DSQFakeQuantize':       DSQFakeQuantize,        # DSQ                          # noqa: E241
    'PACTFakeQuantize':      PACTFakeQuantize,       # PACT                         # noqa: E241
    'TqtFakeQuantize':       TqtFakeQuantize,        # TQT                          # noqa: E241
    'AdaRoundFakeQuantize':  AdaRoundFakeQuantize,   # AdaRound                     # noqa: E241
    'QDropFakeQuantize':     QDropFakeQuantize,      # BRECQ & QDrop                # noqa: E241
}

def create_quantizer(extra_qparams: Dict):
    w_observer = extra_qparams.get('w_observer', None)
    if w_observer:
        assert w_observer in ObserverDict, \
            'Do not support observer name: {}'.format(w_observer)
        w_observer = ObserverDict[w_observer]
    
    # 处理激活量化可选配置
    a_observer = extra_qparams.get('a_observer', None)
    a_fakequantize = extra_qparams.get('a_fakequantize', None)
    a_qscheme = extra_qparams.get('a_qscheme', None)
    
    if a_observer:
        assert a_observer in ObserverDict, \
            'Do not support observer name: {}'.format(a_observer)
        a_observer = ObserverDict[a_observer]
    
    w_fakequantize = extra_qparams.get('w_fakequantize', None)
    if w_fakequantize:
        assert w_fakequantize in FakeQuantizeDict, \
            'Do not support fakequantize name: {}'.format(w_fakequantize)
        w_fakequantize = FakeQuantizeDict[w_fakequantize]
    
    if a_fakequantize:
        assert a_fakequantize in FakeQuantizeDict, \
            'Do not support fakequantize name: {}'.format(a_fakequantize)
        a_fakequantize = FakeQuantizeDict[a_fakequantize]

    w_qscheme = QuantizeScheme(**extra_qparams['w_qscheme'])
    w_fakeq_params = extra_qparams.get('w_fakeq_params', {})
    a_fakeq_params = extra_qparams.get('a_fakeq_params', {})
    
    print('Weight Qconfig:\n    FakeQuantize: {} Params: {}\n'
          '    Oberver:      {} Params: {}'.format(w_fakequantize.__name__, w_fakeq_params,
                                                   w_observer.__name__, str(w_qscheme)))
    
    # 创建权重量化器
    w_fakequantize_instance = w_fakequantize(w_observer,**w_qscheme.to_observer_params(),**w_fakeq_params)
    
    # 创建激活量化器（如果启用）
    if a_qscheme and a_observer and a_fakequantize:
        a_qscheme = QuantizeScheme(**a_qscheme)
        print('Activation Qconfig:\n    FakeQuantize: {} Params: {}\n'
              '    Oberver:      {} Params: {}'.format(a_fakequantize.__name__, a_fakeq_params,
                                                       a_observer.__name__, str(a_qscheme)))
        a_fakequantize_instance = a_fakequantize(a_observer,**a_qscheme.to_observer_params(),**a_fakeq_params)
    else:
        print('Activation Qconfig: Disabled (W-only quantization)')
        a_fakequantize_instance = nn.Identity()
    
    return w_fakequantize_instance, a_fakequantize_instance


def get_to_be_replaced_modules(model):
    modules = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            module = ModuleHook(name, m)
            modules.append(module)
        elif isinstance(m, nn.Linear):
            module = ModuleHook(name, m)
            modules.append(module)
    return modules

def quant_model(model, qconfig_dict, use_gptq=False, gptq_config=None):
    """
    量化模型
    
    Args:
        model: 要量化的模型
        qconfig_dict: 量化配置字典
        use_gptq: 是否使用GPTQ量化
        gptq_config: GPTQ配置参数
    """
    wrapped_modules = []
    
    if use_gptq:
        # 使用GPTQ量化
        print("使用GPTQ量化...")
        if gptq_config is None:
            gptq_config = {
                'bits': 8,
                'perchannel': True,
                'sym': True,
                'blocksize': 128,
                'percdamp': 0.01
            }
        
        module_dict = {}
        for name, m in model.named_modules():
            module_dict[name] = m
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")
            
            if isinstance(m, nn.Conv2d):
                # 使用GPTQ卷积层
                idx = idx + 1 if idx != 0 else idx
                new_m = GPTQConv2d(
                    m.in_channels, m.out_channels, m.kernel_size,
                    m.stride, m.padding, m.dilation,
                    m.groups, m.bias is not None,
                    bits=gptq_config['bits'],
                    perchannel=gptq_config['perchannel'],
                    sym=gptq_config['sym']
                )
                new_m.weight.data.copy_(m.weight.data)
                if m.bias is not None:
                    new_m.bias = nn.Parameter(m.bias.detach().clone())
                else:
                    new_m.bias = None
                setattr(father_module, name[idx:], new_m)
                module = ModuleHook(name, new_m)
                wrapped_modules.append(module)
            elif isinstance(m, nn.Linear) and not isinstance(m, NonDynamicallyQuantizableLinear):
                # 使用GPTQ线性层
                idx = idx + 1 if idx != 0 else idx
                new_m = GPTQLinear(
                    m.in_features, m.out_features, 
                    bias=m.bias is not None,
                    bits=gptq_config['bits'],
                    perchannel=gptq_config['perchannel'],
                    sym=gptq_config['sym']
                )
                new_m.weight.data.copy_(m.weight.data)
                if m.bias is not None:
                    new_m.bias = nn.Parameter(m.bias.detach().clone())
                else:
                    new_m.bias = None
                setattr(father_module, name[idx:], new_m)
                module = ModuleHook(name, new_m)
                wrapped_modules.append(module)
    else:
        # 使用原有的MQBench量化
        print("使用MQBench量化...")
        w_quantizer, a_quantizer = create_quantizer(qconfig_dict)
        module_dict = {}
        for name, m in model.named_modules():
            module_dict[name] = m
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")
            
            if isinstance(m, nn.Conv2d):
                # 卷积层量化
                idx = idx + 1 if idx != 0 else idx
                new_m = QuantConv2d(
                    m.in_channels, m.out_channels, m.kernel_size,
                    m.stride, m.padding, m.dilation,
                    m.groups, m.bias is not None,
                    w_quantizer, a_quantizer
                )
                new_m.weight.data = m.weight.data
                new_m.bias = m.bias
                setattr(father_module, name[idx:], new_m)
                module = ModuleHook(name, new_m)
                wrapped_modules.append(module)
            elif isinstance(m, nn.Linear) and not isinstance(m, NonDynamicallyQuantizableLinear):
                # 线性层量化
                idx = idx + 1 if idx != 0 else idx
                new_m = QuantLinear(m.in_features, m.out_features, w_quantizer, a_quantizer)
                new_m.weight.data = m.weight.data
                new_m.bias = m.bias
                setattr(father_module, name[idx:], new_m)
                module = ModuleHook(name, new_m)
                wrapped_modules.append(module)
    
    return wrapped_modules


def collect_gptq_data(model, dataloader, num_samples=128):
    """
    收集GPTQ校准数据

    Args:
        model: 模型
        dataloader: 数据加载器
        num_samples: 样本数量
    """
    print(f"收集GPTQ校准数据，样本数: {num_samples}")

    # 使用一个字典来跟踪每个模块处理的样本数
    module_sample_counts = {name: 0 for name, mod in model.named_modules() if isinstance(mod, (GPTQConv2d, GPTQLinear))}

    # quant_model.py (修改后的逻辑)

    def hook_fn(name, module):
        def hook(mod, inp, out):
            # 检查当前模块是否还需要收集数据
            if module_sample_counts.get(name, num_samples) < num_samples:
                if hasattr(mod, 'add_batch'):
                    # 关键：forward_hook 的输入尚未经过模块内的 a_quantizer，这里手动量化以对齐推理轨迹
                    x = inp[0].detach()
                    # 关键：用本层 a_quantizer 模拟推理时的量化输入
                    if hasattr(mod, 'a_quantizer') and mod.a_quantizer.ready():
                        with torch.no_grad():
                            x = mod.a_quantizer.quantize(x)
                    mod.add_batch(x, out)
                    # ------------------

                    # 更新该模块的样本计数
                    module_sample_counts[name] += 1

        return hook

    # 注册钩子
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (GPTQConv2d, GPTQLinear)):
            hook = module.register_forward_hook(hook_fn(name, module))
            hooks.append(hook)

    # 运行数据
    device = next(model.parameters()).device
    total_processed = 0
    with torch.no_grad():
        for batch in dataloader:
            # 检查是否所有模块都已收集足够样本
            if all(count >= num_samples for count in module_sample_counts.values()):
                break

            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device)
            model(inputs)

            total_processed += 1
            if total_processed % 10 == 0:
                print(f"已处理 {total_processed} 个样本")

    # 移除钩子
    for hook in hooks:
        hook.remove()

    print(f"数据收集完成，共处理 {total_processed} 个样本")

def execute_gptq_quantization(model, gptq_config=None):
    """
    执行GPTQ量化

    Args:
        model: 模型
        gptq_config: GPTQ配置参数

    Returns:
        dict: 每个量化层的Hessian诊断信息
    """
    if gptq_config is None:
        gptq_config = {
            'blocksize': 128,
            'percdamp': 0.01,
            'use_cholesky': True,
            'handle_dead_neurons': True
        }

    print("执行GPTQ量化...")
    print(f"GPTQ配置: blocksize={gptq_config.get('blocksize', 128)}, "
          f"percdamp={gptq_config.get('percdamp', 0.01)}, "
          f"use_cholesky={gptq_config.get('use_cholesky', True)}, "
          f"handle_dead_neurons={gptq_config.get('handle_dead_neurons', True)}")

    # 【新增】用于存储所有诊断信息的字典
    all_diagnostics = {}

    for name, module in model.named_modules():
        if isinstance(module, (GPTQConv2d, GPTQLinear)):
            print(f"量化层: {name}")

            diagnostics = None  # 初始化
            # 检查模块是否有扩展的GPTQ方法
            if hasattr(module, 'gptq_quantize_with_options'):
                # (此部分为示例，当前代码未实现，但为保持逻辑完整性而保留)
                diagnostics = module.gptq_quantize_with_options(
                    blocksize=gptq_config.get('blocksize', 128),
                    percdamp=gptq_config.get('percdamp', 0.01),
                    use_cholesky=gptq_config.get('use_cholesky', True),
                    handle_dead_neurons=gptq_config.get('handle_dead_neurons', True)
                )
            else:
                # 使用原有的GPTQ方法
                diagnostics = module.gptq_quantize(
                    blocksize=gptq_config.get('blocksize', 128),
                    percdamp=gptq_config.get('percdamp', 0.01)
                )

            # 【新增】存储该层的诊断信息
            if diagnostics:
                all_diagnostics[name] = diagnostics

    print("GPTQ量化完成!")
    # 【新增】返回收集到的所有诊断信息
    return all_diagnostics

