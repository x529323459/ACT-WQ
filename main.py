import time
import torch
import argparse
import os
import copy
import yaml
from rtdetr.model_solver import config_model,config_solver
from mqbench.quant_model import *
from mqbench.utils.state import *
from mqbench.recon_model import recon_model
from utils.fuse import fuse_model
from mqbench.gptq_act_calibration import calibrate_gptq_activations


qconfig = {
    'w_observer': 'MinMaxObserver',  # custom weight observer
    'a_observer': 'EMAMSEObserver',  # custom activation observer
    'w_fakequantize': 'NNIEFakeQuantize',  # custom weight fake quantize function
    'a_fakequantize': 'NNIEFakeQuantize',  # custom activation fake quantize function
    # 'w_fakequantize': 'AdaRoundFakeQuantize',  # custom weight fake quantize function
    # 'a_fakequantize': 'AdaRoundFakeQuantize',  # custom activation fake quantize function
    'w_qscheme': {
        'bit': 8,  # custom bitwidth for weight,
        'symmetry': True,  # custom whether quant is symmetric for weight,
        'per_channel': False,  # custom whether quant is per-channel or per-tensor for weight,
        'pot_scale': False,  # custom whether scale is power of two for weight.
    },
    'a_qscheme': {
        'bit': 8,  # custom bitwidth for activation,
        'symmetry': True,  # custom whether quant is symmetric for activation,
        'per_channel': False,  # custom whether quant is per-channel or per-tensor for activation,
        'pot_scale': False,  # custom whether scale is power of two for activation.
    }
}

# GPTQ量化配置
gptq_config = {
    'bits': 8,  # 量化位数
    'perchannel': True,  # 按通道量化
    'sym': True,  # 对称量化
    'blocksize': 128,  # 分块大小
    'percdamp': 0.01,  # 阻尼百分比
    'use_gptq': True  # 是否使用GPTQ量化
}

def main(args):
    print("config rtdetr model based on config...")
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

    print("import pre_trained weight from resume...")
    if cfg['resume']:  # 是否导入已有模型权重
        path = cfg['resume']
        print(f'resume from {path}')
        solver.resume(path)

    # 根据量化config进行网络量化算子更替
    fp_model = solver.ema.module if solver.ema else solver.model
    fuse_model(fp_model)# 进行算子融合 研究内容===
    q_model = copy.deepcopy(fp_model)

    # 选择量化方式
    if gptq_config['use_gptq']:
        print("使用GPTQ量化...")
        # 完成网络的封装，使用GPTQ量化
        wrapped_modules = quant_model(q_model, qconfig, use_gptq=True, gptq_config=gptq_config)

        # 设置solver（初始化output_dir等属性）
        solver.setup(q_model)

        # advanced_ptq_rtdetr.py (修改后的逻辑)

        # 第 1 步：先校准激活
        if args.gptq_act:
            print("步骤 1: 执行 GPTQ 激活量化校准 ...")
            calibrate_gptq_activations(
                fp_model=fp_model,
                q_model=q_model,
                dataloader=solver.calib_dataloader,
                act_bits=args.gptq_act_bits,
                num_batches = args.gptq_act_batches,
                per_batch_samples = args.gptq_act_samples,
                max_channels = args.gptq_act_max_channels,
                device='cuda'
            )
            print("激活校准完成。")

        # 第 2 步：再收集权重校准数据（此时 a_quantizer.ready() 为 True）
        print("步骤 2: 收集 GPTQ 权重校准数据 ...")
        collect_gptq_data(q_model, solver.calib_dataloader, num_samples=128)
        print("权重数据收集完成。")

        # 第 3 步：最后执行权重量化
        print("步骤 3: 执行 GPTQ 权重量化 ...")
        execute_gptq_quantization(q_model, gptq_config)
        print("权重量化完成。")

        # 启用量化并评估
        enable_quantization(q_model)

        # 启用量化（此时GPTQ层已在forward中对权重+激活进行量化）
        enable_quantization(q_model)

    else:
        print("使用MQBench量化...")
        # 完成网络的封装，使用MQBench量化
        wrapped_modules = quant_model(q_model, qconfig, use_gptq=False)

        # 进行网络校准
        solver.setup(q_model)
        enable_calibration(q_model)
        solver.calib()

        # 是否重构
        if args.reconstruction:
            recon_model(solver, q_model, fp_model)


    # 进行测试
    torch.save(q_model.state_dict(), 'output/Adaround/model_adaround20000_weight.pth')

    # 新增：统计一次验证（完整 val 一轮）的耗时
    start_time = time.time()
    solver.val()
    end_time = time.time()
    print(f"本次验证一轮耗时: {end_time - start_time:.2f} 秒")

    from decoder_similarity import compute_decoder_similarity

    # 使用 solver.calib_dataloader 作为统计数据源
    layer_names, sims = compute_decoder_similarity(
        fp_model=fp_model,
        q_model=q_model,
        dataloader=solver.calib_dataloader,
        device='cuda',
        max_batches=50,   # 可以视显存/时间调整
        mode='cos',       # 或 'mse'
    )

    # 如需保存到文件，方便后处理/画图：
    import json
    os.makedirs('output/decoder_stats', exist_ok=True)
    with open('output/decoder_stats/decoder_cos_sim.json', 'w') as f:
        json.dump({
            "layers": layer_names,
            "cosine": sims,
        }, f, indent=2)
    return fp_model, q_model, solver

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='rtdetr/config.yml')
    parser.add_argument('--resume', '-r', type=str, default='pre_model/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth')
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--reconstruction', '-rec', type=str, default= True)
    parser.add_argument('--test-only', action='store_true', default=True)
    parser.add_argument('--amp', action='store_true', default=False, )
    parser.add_argument('--seed', type=int, help='seed', default = 3)
    parser.add_argument('--use-gptq', action='store_true', help='使用GPTQ量化')
    parser.add_argument('--gptq-bits', type=int, default=6, help='GPTQ量化位数')
    parser.add_argument('--gptq-blocksize', type=int, default=128, help='GPTQ分块大小')
    parser.add_argument('--gptq-percdamp', type=float, default=0.01, help='GPTQ阻尼百分比')
    parser.add_argument('--gptq-act', action='store_true', default=True, help='为GPTQ启用激活量化')
    parser.add_argument('--gptq-act-bits', type=int, default=6, help='GPTQ激活量化位宽')
    parser.add_argument('--gptq-act-batches', type=int, default=256, help='用于激活校准的批次数（建议≥256）')
    parser.add_argument('--gptq-act-samples', type=int, default=16384, help='每批每层采样激活元素数（建议≥8k）')
    parser.add_argument('--gptq-act-max-channels', type=int, default=-1,help='每层最大采样通道数，-1表示不限（覆盖全通道）')
    args = parser.parse_args()
    
    # 更新GPTQ配置
    if args.use_gptq:
        gptq_config['use_gptq'] = True
        gptq_config['bits'] = args.gptq_bits
        gptq_config['blocksize'] = args.gptq_blocksize
        gptq_config['percdamp'] = args.gptq_percdamp

    main(args)



