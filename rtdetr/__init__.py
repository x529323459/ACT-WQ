# RTDetr 模块初始化文件
"""
RTDetr (Real-Time Detection Transformer) 模块

这个包包含了RTDetr模型的完整实现，包括：
- 模型架构 (nn/)
- 数据处理 (data/)
- 优化器 (optim/)
- 工具函数 (utils/)
- 其他辅助功能 (misc/)
"""

__version__ = "1.0.0"
__author__ = "RTDetr Team"

# 导入主要组件
from .model_solver import config_model, config_solver, config_optimizer

# 导入核心模型
from .nn.rtdetr import RTDETR
from .nn.presnet import PResNet
from .nn.hybrid_encoder import HybridEncoder
from .nn.rtdetr_decoder import RTDETRTransformer
from .nn.rtdetr_criterion import SetCriterion
from .nn.rtdetr_postprocessor import RTDETRPostProcessor
from .nn.matcher import HungarianMatcher

# 导入数据处理组件
from .data.transforms import Compose
from .data.coco.coco_dataset import CocoDetection
from .data.dataloader import DataLoader, default_collate_fn

# 导入优化组件
from .optim.ema import ModelEMA

# 导入工具
from .utils.solver import Solver

__all__ = [
    # 配置函数
    'config_model', 'config_solver', 'config_optimizer',
    
    # 核心模型
    'RTDETR', 'PResNet', 'HybridEncoder', 'RTDETRTransformer',
    'SetCriterion', 'RTDETRPostProcessor', 'HungarianMatcher',
    
    # 数据处理
    'Compose', 'CocoDetection', 'DataLoader', 'default_collate_fn',
    
    # 优化组件
    'ModelEMA',
    
    # 工具
    'Solver'
]
