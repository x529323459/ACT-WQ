import torch
import torch.nn as nn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from ACT_WQ.quant_modules import GPTQConv2d, GPTQLinear


def quant_model(model, qconfig_dict=None, use_gptq=True, gptq_config=None):
    """
    Replace Conv2d/Linear with GPTQ variants.
    Returns a list of replaced module names.
    """
    if not use_gptq:
        raise ValueError("This repository is trimmed to GPTQ-only quantization.")

    if gptq_config is None:
        gptq_config = {
            "bits": 8,
            "perchannel": True,
            "sym": True,
            "blocksize": 128,
            "percdamp": 0.01,
        }

    replaced = []
    module_dict = {}
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind(".")
        idx = 0 if idx == -1 else idx
        father_name = name[:idx]
        if father_name not in module_dict:
            raise RuntimeError(f"father module {father_name} not found")
        father_module = module_dict[father_name]

        if isinstance(m, nn.Conv2d):
            idx = idx + 1 if idx != 0 else idx
            new_m = GPTQConv2d(
                m.in_channels,
                m.out_channels,
                m.kernel_size,
                m.stride,
                m.padding,
                m.dilation,
                m.groups,
                m.bias is not None,
                bits=gptq_config["bits"],
                perchannel=gptq_config["perchannel"],
                sym=gptq_config["sym"],
            )
            new_m.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                new_m.bias = nn.Parameter(m.bias.detach().clone())
            else:
                new_m.bias = None
            setattr(father_module, name[idx:], new_m)
            replaced.append(name)

        elif isinstance(m, nn.Linear) and not isinstance(m, NonDynamicallyQuantizableLinear):
            idx = idx + 1 if idx != 0 else idx
            new_m = GPTQLinear(
                m.in_features,
                m.out_features,
                bias=m.bias is not None,
                bits=gptq_config["bits"],
                perchannel=gptq_config["perchannel"],
                sym=gptq_config["sym"],
            )
            new_m.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                new_m.bias = nn.Parameter(m.bias.detach().clone())
            else:
                new_m.bias = None
            setattr(father_module, name[idx:], new_m)
            replaced.append(name)

    return replaced


def collect_gptq_data(model, dataloader, num_samples=128):
    print(f"Collecting GPTQ calibration data, samples per layer: {num_samples}")

    module_sample_counts = {
        name: 0
        for name, mod in model.named_modules()
        if isinstance(mod, (GPTQConv2d, GPTQLinear))
    }

    def hook_fn(name, module):
        def hook(mod, inp, out):
            if module_sample_counts.get(name, num_samples) < num_samples:
                if hasattr(mod, "add_batch"):
                    x = inp[0].detach()
                    if hasattr(mod, "a_quantizer") and mod.a_quantizer.ready():
                        with torch.no_grad():
                            x = mod.a_quantizer.quantize(x)
                    mod.add_batch(x, out)
                    module_sample_counts[name] += 1

        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (GPTQConv2d, GPTQLinear)):
            hook = module.register_forward_hook(hook_fn(name, module))
            hooks.append(hook)

    device = next(model.parameters()).device
    total_processed = 0
    with torch.no_grad():
        for batch in dataloader:
            if all(count >= num_samples for count in module_sample_counts.values()):
                break

            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device)
            model(inputs)

            total_processed += 1
            if total_processed % 10 == 0:
                print(f"Processed {total_processed} batches")

    for hook in hooks:
        hook.remove()

    print(f"GPTQ data collection completed, processed {total_processed} batches")


def execute_gptq_quantization(model, gptq_config=None):
    if gptq_config is None:
        gptq_config = {
            "blocksize": 128,
            "percdamp": 0.01,
            "use_cholesky": True,
            "handle_dead_neurons": True,
        }

    print("Running GPTQ weight quantization...")
    print(
        f"GPTQ config: blocksize={gptq_config.get('blocksize', 128)}, "
        f"percdamp={gptq_config.get('percdamp', 0.01)}, "
        f"use_cholesky={gptq_config.get('use_cholesky', True)}, "
        f"handle_dead_neurons={gptq_config.get('handle_dead_neurons', True)}"
    )

    all_diagnostics = {}
    for name, module in model.named_modules():
        if isinstance(module, (GPTQConv2d, GPTQLinear)):
            print(f"Quantizing layer: {name}")

            diagnostics = None
            if hasattr(module, "gptq_quantize_with_options"):
                diagnostics = module.gptq_quantize_with_options(
                    blocksize=gptq_config.get("blocksize", 128),
                    percdamp=gptq_config.get("percdamp", 0.01),
                    use_cholesky=gptq_config.get("use_cholesky", True),
                    handle_dead_neurons=gptq_config.get("handle_dead_neurons", True),
                )
            else:
                diagnostics = module.gptq_quantize(
                    blocksize=gptq_config.get("blocksize", 128),
                    percdamp=gptq_config.get("percdamp", 0.01),
                )

            if diagnostics:
                all_diagnostics[name] = diagnostics

    print("GPTQ quantization finished")
    return all_diagnostics
