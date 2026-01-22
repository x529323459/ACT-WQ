import argparse
import os
import torch
import yaml
import copy
import re
import numpy as np

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .nn.presnet import PResNet
from .nn.hybrid_encoder import HybridEncoder
from .nn.rtdetr import RTDETR
from .nn.rtdetr_decoder import RTDETRTransformer
from .nn.matcher import HungarianMatcher
from .nn.rtdetr_criterion import SetCriterion
from .nn.rtdetr_postprocessor import RTDETRPostProcessor
from .optim.ema import ModelEMA
from .data.transforms import Compose
from .data.coco.coco_dataset import CocoDetection
from .data.dataloader import DataLoader,default_collate_fn
from .utils.solver import Solver  # hy修改
from torch.cuda.amp.grad_scaler import GradScaler


def config_model(cfg):
    # 配置网络模型
    # backbone: PResNet
    # encoder: HybridEncoder
    # decoder: RTDETRTransformer
    # multi_scale
    model_params = cfg['model']
    params = model_params['PResNet']
    backbone = PResNet(**params)
    params = model_params['HybridEncoder']
    encoder = HybridEncoder(**params)
    params = model_params['RTDETRTransformer']
    decoder = RTDETRTransformer(**params)
    multi_scale = model_params['multi_scale']
    model = RTDETR(backbone, encoder, decoder, multi_scale)
    return model

# def resume_model(cfg, model):
#     # 加载预训练权重
#     if cfg['use_ema']:
#         params = cfg['ema']
#         ema = ModelEMA(model, **params)
#     if cfg['resume']:
#         path = cfg['resume']
#         print(f'resume from {path}')
#         state = torch.load(path, map_location='cpu')
#         if 'ema' in state:
#             ema.load_state_dict(state['ema'])
#             print('Loading ema.state_dict')
#             model = ema.module
#     return model, ema

def config_optimizer(cfg,model):
    # 配置数据读取
    params = cfg['optimizer']
    adam_params = get_optim_params(params, model)
    optimizer = optim.AdamW(adam_params)
    params = cfg['lr_scheduler']
    lr = lr_scheduler.MultiStepLR(optimizer,**params)
    return optimizer, lr

def config_solver(cfg,model):
    model_params = cfg['model']
    params = model_params['SetCriterion']['matcher']
    matcher = HungarianMatcher(**params)
    params = model_params['SetCriterion']
    params['matcher'] = matcher
    critertion = SetCriterion(**params)
    params = model_params['RTDETRPostProcessor']
    postprocessor = RTDETRPostProcessor(**params)

    optimizer, lr_scheduler = config_optimizer(cfg, model) #优化器和lr的配置要在resume之前
    # model,ema = resume_model(cfg, model)
    if cfg['use_ema']:
        params = cfg['ema']
        ema = ModelEMA(model, **params)
    # train_dataloader配置
    params = cfg['dataloader']['train_dataloader']['dataset']['transforms']
    transforms = Compose(**params)
    img_folder = cfg['dataloader']['train_dataloader']['dataset']['img_folder']
    ann_file = cfg['dataloader']['train_dataloader']['dataset']['ann_file']
    train_coco_dataset = CocoDetection(transforms,img_folder = img_folder, ann_file = ann_file,
                                 return_masks=False,remap_mscoco_category = True)
    print(train_coco_dataset[0])
    params = cfg['dataloader']['train_dataloader']
    # 配置dataloader参数，特殊处理
    if params['collate_fn'] == 'default_collate_fn':
        collate_fn = default_collate_fn
    train_dataloader = DataLoader(dataset=train_coco_dataset,
                                batch_size=params['batch_size'],
                                num_workers=params['num_workers'],
                                drop_last=params['drop_last'],
                                collate_fn=collate_fn)
    train_dataloader.shuffle = params['shuffle']
    
    val_dataloader = None
    calib_dataloader = None
    # 配置calib_dataloader
    params = cfg['dataloader']['calib_dataloader']['dataset']['transforms']
    transforms = Compose(**params)
    img_folder = cfg['dataloader']['calib_dataloader']['dataset']['img_folder']
    ann_file = cfg['dataloader']['calib_dataloader']['dataset']['ann_file']
    calib_coco_dataset = CocoDetection(transforms, img_folder=img_folder, ann_file=ann_file,
                                       return_masks=False, remap_mscoco_category=False)
    params = cfg['dataloader']['calib_dataloader']
    if params['collate_fn'] == 'default_collate_fn':
        collate_fn = default_collate_fn
    calib_dataloader = DataLoader(dataset=calib_coco_dataset,
                                  batch_size=params['batch_size'],
                                  num_workers=params['num_workers'],
                                  drop_last=params['drop_last'],
                                  collate_fn=collate_fn)
    calib_dataloader.shuffle = False

    # 配置val_dataloader
    params = cfg['dataloader']['val_dataloader']['dataset']['transforms']
    transforms = Compose(**params)
    img_folder = cfg['dataloader']['val_dataloader']['dataset']['img_folder']
    ann_file = cfg['dataloader']['val_dataloader']['dataset']['ann_file']
    val_coco_dataset = CocoDetection(transforms, img_folder=img_folder, ann_file=ann_file,
                                       return_masks=False, remap_mscoco_category=False)
    params = cfg['dataloader']['val_dataloader']
    # 配置dataloader参数，特殊处理
    if params['collate_fn'] == 'default_collate_fn':
        collate_fn = default_collate_fn
    val_dataloader = DataLoader(dataset = val_coco_dataset,
                                batch_size = params['batch_size'],
                                num_workers = params['num_workers'],
                                drop_last = params['drop_last'],
                                collate_fn = collate_fn)
    val_dataloader.shuffle = params['shuffle']

    solver = Solver(cfg, model, critertion, postprocessor, ema,
                    optimizer, lr_scheduler, train_dataloader, val_dataloader,calib_dataloader)
    return solver

def get_optim_params(cfg, model):
    '''
    E.g.:
        ^(?=.*a)(?=.*b).*$         means including a and b
        ^((?!b.)*a((?!b).)*$       means including a but not b
        ^((?!b|c).)*a((?!b|c).)*$  means including a but not (b | c)
    '''
    cfg = copy.deepcopy(cfg)

    if 'params' not in cfg:
        return model.parameters()

    assert isinstance(cfg['params'], list), ''

    param_groups = []
    visited = []
    for pg in cfg['params']:
        pattern = pg['params']
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
        pg['params'] = params.values()
        param_groups.append(pg)
        visited.extend(list(params.keys()))

    names = [k for k, v in model.named_parameters() if v.requires_grad]

    if len(visited) < len(names):
        unseen = set(names) - set(visited)
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
        param_groups.append({'params': params.values()})
        visited.extend(list(params.keys()))

    assert len(visited) == len(names), ''

    return param_groups

def main(args):
    _, ext = os.path.splitext(args.config)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"

    with open(args.config, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
        if cfg is None:
            return {}
    cfg['tuning'] = args.tuning
    cfg['resume'] = args.resume
    model = config_model(cfg)
    solver = config_solver(cfg, model)
    if cfg['resume']: #是否导入已有模型权重
        path = cfg['resume']
        print(f'resume from {path}')
        solver.resume(path)
    solver.setup() #配置基本实验数据

    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config.yml')
    parser.add_argument('--resume', '-r', type=str, default='../pre_model/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth' )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=True, )
    parser.add_argument('--amp', action='store_true', default=False, )
    parser.add_argument('--seed', type=int, help='seed', )
    args = parser.parse_args()

    main(args)



    # path = '../../data/coco/val2017/'
    # file = open('../../data/coco/val2017.txt','w')
    # for filename in os.listdir(path):
    #     if (filename.endswith('.jpg')):
    #         print(filename)
    #         file.write(path+filename)
    #         file.write("\n")





