"""No-search activation control using the Full arm's exact sampled inputs."""
from pathlib import Path
import torch
from ACT_WQ import act_calibration as cal
from ACT_WQ.sensitivity_calibration import dump, sample_fingerprint, tensor_hash


def parameters(records, bits, perchannel):
    if not 2 <= bits <= 16 or not records:
        raise ValueError('Min-Max requires valid bit width and nonempty records')
    result = {}
    for name, rows in records.items():
        channels = rows[0]['full_channels']
        lows, highs = [], []
        for c in range(channels):
            parts = [r['x'][r['channel_ids'].index(c)].detach().float().flatten()
                     for r in rows if c in r['channel_ids']]
            if not parts or any(not v.numel() or not torch.isfinite(v).all() for v in parts):
                raise ValueError('Min-Max requires finite samples for every channel: ' + name)
            lows.append(min(float(v.min()) for v in parts))
            highs.append(max(float(v.max()) for v in parts))
        if not perchannel:
            lows, highs = [min(lows)], [max(highs)]
        lo = torch.tensor(lows).clamp(max=0)
        hi = torch.tensor(highs).clamp(min=0)
        scale = ((hi - lo) / (2**bits - 1)).clamp(min=1e-8)
        zero = (-lo / scale).round().clamp(0, 2**bits - 1)
        result[name] = dict(scale=scale.tolist(), zero=zero.tolist(),
            clip=((2**bits - 1 - zero) * scale).tolist(), clip_min=(-zero * scale).tolist())
    return result


def calibrate_minmax(model, *, sample_cache, output_dir, act_bits, perchannel):
    if not sample_cache:
        raise ValueError('Paired Min-Max requires the Full activation sample cache')
    payload = torch.load(str(sample_cache), map_location='cpu')
    records, batches = payload['records'], payload['batches']
    layers = cal._get_gptq_layers(model)
    if not batches or set(records) - set(layers):
        raise ValueError('Invalid Min-Max sample cache/model layer coverage')
    before = {n: tensor_hash(p) for n, p in model.named_parameters()}
    params = parameters(records, act_bits, perchannel)
    for name, layer in layers.items():
        layer.enable_activation_quant(bits=act_bits, symmetric=False, perchannel=perchannel,
            ch_axis=1 if isinstance(layer, cal.GPTQConv2d) else -1)
        if name in params:
            layer.set_activation_qparams(**params[name])
    if before != {n: tensor_hash(p) for n, p in model.named_parameters()}:
        raise RuntimeError('Min-Max calibration changed model parameters')
    summary = dict(mode='Min-Max; entire activation clipping-search module removed',
        granularity='channel' if perchannel else 'tensor', gradient_used=False,
        clipping_search=False, candidate_policy='sample_minmax',
        sample_source=str(Path(sample_cache).resolve()), sample_fingerprint=sample_fingerprint(records),
        initialized_layers=len(params), images=sum(b['images'] for b in batches), batches=len(batches),
        parameters_unchanged_during_collection=True,
        range_definition='sample extrema including zero; affine integer zero point; no percentile or MSE search')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'activation_params.json', params)
    dump(out / 'calibration_summary.json', summary)
    print('MINMAX complete:', len(params), 'layers; no gradient or clipping search', flush=True)
    return params, summary
