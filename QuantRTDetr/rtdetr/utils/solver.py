"""by lyuwenyu
    modified by YingHuo
"""
import json
import time

import torch 
import torch.nn as nn 

from datetime import datetime,timedelta
from pathlib import Path 
from typing import Dict

from ..misc import dist
from torch.cuda.amp.grad_scaler import GradScaler
from ..data import get_coco_api_from_dataset
from .det_engine import train_one_epoch, evaluate,run_calib


class Solver(object):
    def __init__(self, cfg, model, criterion, postprocessor,ema,
                 optimizer, lr_scheduler, train_dataloader, val_dataloader,calib_dataloader) -> None:
        self.cfg = cfg
        self.model = model
        self.find_unused_parameters = cfg['find_unused_parameters']
        self.sync_bn = cfg['sync_bn']
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.ema = ema
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.calib_dataloader = calib_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.clip_max_norm = cfg['clip_max_norm']
        self.epoches = cfg['epoches']
        self.log_step = 10
        self.checkpoint_step = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # 设置输出目录
        self.output_dir = Path(cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self, new_model):
        '''Avoid instantiating unnecessary classes 
        '''
        self.last_epoch = -1
        self.model = dist.warp_model(new_model.to(self.device), self.find_unused_parameters, self.sync_bn)
        self.criterion = self.criterion.to(self.device)

        # NOTE (lvwenyu): should load_tuning_state before ema instance building
        if self.cfg.get('tuning'):
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)
       # if self.cfg['scaler']:
        if self.cfg['use_amp'] and torch.cuda.is_available():
            self.scaler = GradScaler()
        else:
            self.scaler = None
        self.ema = self.ema.to(self.device) if self.ema is not None else None

        self.output_dir = Path(self.cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def state_dict(self, last_epoch):
        '''state dict
        '''
        state = {}
        state['model'] = dist.de_parallel(self.model).state_dict()
        state['date'] = datetime.now().isoformat()

        # TODO
        state['last_epoch'] = last_epoch

        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            # state['last_epoch'] = self.lr_scheduler.last_epoch

        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()

        return state


    def load_state_dict(self, state):
        '''load state dict
        '''
        # TODO
        if getattr(self, 'last_epoch', None) and 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')

        if getattr(self, 'model', None) and 'model' in state:
            if dist.is_parallel(self.model):
                self.model.module.load_state_dict(state['model'])
            else:
                self.model.load_state_dict(state['model'])
            print('Loading model.state_dict')

        if getattr(self, 'ema', None) and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            print('Loading ema.state_dict')

        if getattr(self, 'optimizer', None) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')

        if getattr(self, 'lr_scheduler', None) and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')

        if getattr(self, 'scaler', None) and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')


    def save(self, path):
        '''save state
        '''
        state = self.state_dict()
        dist.save_on_master(state, path)


    def resume(self, path):
        '''load resume
        '''
        # for cuda:0 memory
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state)

    def load_tuning_state(self, path,):
        """only load model for tuning and skip missed/dismatched keys
        """
        if 'http' in path:
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        module = dist.de_parallel(self.model)
        
        # TODO hard code
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

    def calib(self, ):  # 校准阶段 hy补充
        print("Start calib...")
        self.calib_dataloader = dist.warp_loader(self.calib_dataloader, \
                                                 shuffle=self.calib_dataloader.shuffle)  # hy修改
        # base_ds = get_coco_api_from_dataset(self.calib_dataloader.dataset)
       # module = self.ema.module if self.ema else self.model
        start_time = time.time()
        with torch.no_grad():
            calib_stats = run_calib(
                self.model, self.criterion, self.postprocessor,self.calib_dataloader,self.device)
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print('Calib time {}'.format(total_time_str))

    def val(self, ):
        self.val_dataloader = dist.warp_loader(self.val_dataloader, \
                                               shuffle=self.val_dataloader.shuffle)
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        #module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(self.model, self.criterion, self.postprocessor,
                                              self.val_dataloader, base_ds, self.device, self.output_dir)

        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return test_stats

    def fit(self, ):
        print("Start training")
        self.train_dataloader = dist.warp_loader(self.train_dataloader, \
                                                 shuffle=self.train_dataloader.shuffle)  # hy修改
        self.val_dataloader = dist.warp_loader(self.val_dataloader, \
                                               shuffle=self.val_dataloader.shuffle)  # hy修改
        # args = self.cfg
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, self.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                self.clip_max_norm, print_freq=self.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % self.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # TODO
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time))) # hy 改
        print('Training time {}'.format(total_time_str))


