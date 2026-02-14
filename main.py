import argparse
import copy
import os
import time

import torch
import yaml

from ACT_WQ.act_calibration import calibrate_gptq_activations
from ACT_WQ.quant_model import collect_gptq_data, execute_gptq_quantization, quant_model
from rtdetr.model_solver import config_model, config_solver
from utils.fuse import fuse_model


gptq_config = {
    "bits": 4,
    "perchannel": True,
    "sym": True,
    "blocksize": 128,
    "percdamp": 0.01,
    "use_gptq": True,
}


def main(args):
    _, ext = os.path.splitext(args.config)
    assert ext in [".yml", ".yaml"], "only support yaml files"

    with open(args.config, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}

    cfg["tuning"] = args.tuning
    cfg["resume"] = args.resume

    model = config_model(cfg)
    solver = config_solver(cfg, model)

    if cfg.get("resume"):
        solver.resume(cfg["resume"])

    fp_model = solver.ema.module if solver.ema else solver.model
    fuse_model(fp_model)
    q_model = copy.deepcopy(fp_model)

    quant_model(q_model, qconfig_dict=None, use_gptq=True, gptq_config=gptq_config)
    solver.setup(q_model)

    if args.gptq_act:
        print("Step 1: GPTQ activation calibration")
        calibrate_gptq_activations(
            fp_model=fp_model,
            q_model=q_model,
            dataloader=solver.calib_dataloader,
            act_bits=gptq_config["bits"],
            num_batches=args.gptq_act_batches,
            per_batch_samples=args.gptq_act_samples,
            max_channels=args.gptq_act_max_channels,
            device=args.device,
        )

    print("Step 2: Collect GPTQ weight calibration data")
    collect_gptq_data(q_model, solver.calib_dataloader, num_samples=args.gptq_w_samples)

    print("Step 3: Execute GPTQ weight quantization")
    execute_gptq_quantization(q_model, gptq_config)

    if args.save_quant:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "gptq_quant_model.pth")
        torch.save(q_model.state_dict(), save_path)
        print(f"Saved quantized model to: {save_path}")

    start_time = time.time()
    solver.val()
    end_time = time.time()
    print(f"Validation time: {end_time - start_time:.2f}s")

    return fp_model, q_model, solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="rtdetr/config.yml")
    parser.add_argument("--resume", "-r", type=str, default="pre_model/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth")
    parser.add_argument("--tuning", "-t", type=str)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-quant", action="store_true", default=True)
    parser.add_argument("--save-dir", type=str, default="output/gptq")

    parser.add_argument("--gptq-act", action="store_true", default=True)
    parser.add_argument("--gptq-act-batches", type=int, default=256)
    parser.add_argument("--gptq-act-samples", type=int, default=16384)
    parser.add_argument("--gptq-act-max-channels", type=int, default=-1)

    parser.add_argument("--gptq-w-samples", type=int, default=128)

    args = parser.parse_args()

    main(args)
