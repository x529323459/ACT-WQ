from rtdetr.nn.common import ConvNormLayer

import torch.nn as nn
import torch



def fuse_model(model):
    for name, module in model.named_modules():
        if isinstance(module, ConvNormLayer):  # 确保 ConvNormLayer 已被定义
            # 获取 ConvNormLayer 内部的模块
            internal_modules = list(module.modules())[1:]  # 跳过自己，只获取子模块
            if len(internal_modules) == 3 and \
               isinstance(internal_modules[0], nn.Conv2d) and \
               isinstance(internal_modules[1], nn.BatchNorm2d) and \
               isinstance(internal_modules[2], nn.ReLU):
                torch.quantization.fuse_modules(module, ['conv','norm','act'], inplace=True)
            if len(internal_modules) == 3 and \
               isinstance(internal_modules[0], nn.Conv2d) and \
               isinstance(internal_modules[1], nn.BatchNorm2d) and \
               isinstance(internal_modules[2], nn.Identity):
                torch.quantization.fuse_modules(module, ['conv','norm'], inplace=True)
            if len(internal_modules) == 3 and \
               isinstance(internal_modules[0], nn.Conv2d) and \
               isinstance(internal_modules[1], nn.BatchNorm2d) and \
               isinstance(internal_modules[2], nn.SiLU):
                torch.quantization.fuse_modules(module, ['conv','norm'], inplace=True)
        if isinstance(module, nn.Sequential):
            internal_modules = list(module.modules())[1:]
            if len(internal_modules) == 2 and \
                    isinstance(internal_modules[0], nn.Conv2d) and \
                    isinstance(internal_modules[1], nn.BatchNorm2d):
                if 'encoder' in name:
                    torch.quantization.fuse_modules(module, ['0', '1'], inplace=True)
                else:
                    torch.quantization.fuse_modules(module, ['conv', 'norm'], inplace=True)
