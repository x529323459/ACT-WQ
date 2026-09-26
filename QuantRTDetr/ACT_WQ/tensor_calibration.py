"""Production tensor calibration, matching the existing tensor experiment."""
import time
from pathlib import Path
import torch
from ACT_WQ import sensitivity_calibration as sens
from ACT_WQ import act_calibration as cal


def apply_tensor(model, params, bits):
    for name, layer in cal._get_gptq_layers(model).items():
        layer.enable_activation_quant(bits=bits, symmetric=False, perchannel=False,
                                      ch_axis=1 if isinstance(layer, cal.GPTQConv2d) else -1)
        if name in params:
            if any(len(v) != 1 for v in params[name].values()):
                raise RuntimeError('Expected scalar per-tensor parameters: ' + name)
            layer.set_activation_qparams(**params[name])


def calibrate_tensor(fp, model, loader, *, output_dir, mode='sensitivity', sample_cache=None, **kwargs):
    out = Path(output_dir)
    started = time.perf_counter()
    if mode == 'sensitivity':
        _, bootstrap_summary = sens.calibrate_sensitivity(fp, model, loader, output_dir=out, **kwargs)
        payload = torch.load(out / 'original_index_cache.pth', map_location='cpu')
        gradient = torch.load(out / 'original_position_gradients.pth', map_location='cpu')
    elif mode == 'unweighted':
        _, bootstrap_summary = sens.calibrate_unweighted(fp, model, loader, output_dir=out,
                                                         sample_cache=sample_cache, **kwargs)
        payload = torch.load(out / 'unweighted_sample_cache.pth', map_location='cpu')
        gradient = {n: {'g2': None, 'fallback': 'disabled'} for n in payload['records']}
    else:
        raise ValueError('Unsupported per-tensor calibration: ' + mode)
    sens.dump(out / 'channel_bootstrap_summary.json', bootstrap_summary)
    records = payload['records']
    fingerprint = sens.sample_fingerprint(records)
    if fingerprint != bootstrap_summary['sample_fingerprint']:
        raise RuntimeError('Activation sample fingerprint mismatch')
    uniform, sensitive, diagnostic = sens.tensor_search(records, gradient, kwargs['act_bits'])
    params = sensitive if mode == 'sensitivity' else uniform
    apply_tensor(model, params, kwargs['act_bits'])
    summary = {'mode': 'tensor Fisher-inspired' if mode == 'sensitivity' else 'tensor unweighted',
               'granularity': 'tensor', 'sample_fingerprint': fingerprint,
               'initialized_layers': len(params), 'gradient_used': mode == 'sensitivity',
               'gradient_bootstrap': 'shared per-channel bootstrap' if mode == 'sensitivity' else 'disabled',
               'candidate_policy': 'expanded_8bit_v1' if kwargs['act_bits'] == 8 else 'legacy_18',
               'images': sum(b['images'] for b in payload['batches']),
               'batches': len(payload['batches']), 'seconds': time.perf_counter()-started}
    sens.dump(out / 'activation_params.json', params)
    sens.dump(out / 'tensor_search_diagnostics.json', diagnostic)
    sens.dump(out / 'calibration_summary.json', summary)
    print('PER-TENSOR ACTIVATION complete:', len(params), 'layers', flush=True)
    return params, summary
