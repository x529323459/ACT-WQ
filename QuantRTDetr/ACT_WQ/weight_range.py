"""INT8 weight range search for both granularities shared by production and experiments."""
# Baseline first: equal scores retain the original range.
FACTORS = (1.0, .90, .95, .98, 1.02, 1.05, 1.10)


def quantize_layer(layer, enabled, blocksize=128, percdamp=.01):
    """Keep non-INT8 paths on the original solver."""
    original = type(layer).gptq_quantize
    if enabled and int(layer.w_quantizer.maxq.item()) == 255:
        return optimize_layer(layer, original, blocksize=blocksize, percdamp=percdamp)
    return original(layer, blocksize=blocksize, percdamp=percdamp), None


def output_error(weight, reference, hessian):
    """Calibration output error proportional to ||(Q-W)X||_F^2, not weight MSE."""
    error = weight.float().flatten(1) - reference.float().flatten(1)
    return float(((error @ hessian.float()) * error).double().sum().clamp_min(0))


def optimize_layer(layer, original_solver, factors=FACTORS, blocksize=128, percdamp=.01):
    import torch
    if not factors or factors[0] != 1.0 or any(f <= 0 for f in factors):
        raise ValueError('Positive scale factors must start with baseline 1.0')
    if int(layer.w_quantizer.maxq.item()) != 255:
        raise ValueError('Weight range experiment requires 8-bit weights')
    if layer._w_offline_done:
        raise ValueError('Expected original, not previously quantized weights')
    if layer.H is None:
        result = original_solver(layer, blocksize=blocksize, percdamp=percdamp)
        return result, {'skipped': 'no statistics; original solver fallback'}

    reference = layer.weight.detach().clone()
    quantizer = layer.w_quantizer
    old_scale, old_zero, old_maxq = (quantizer.scale.clone(), quantizer.zero.clone(), quantizer.maxq.clone())
    old_done = layer._w_offline_done
    rows = []
    best = None
    try:
        quantizer.find_params(reference.float().flatten(1), weight=True)
        base_scale, base_zero = quantizer.scale.clone(), quantizer.zero.clone()
        for factor in factors:
            layer.weight.data.copy_(reference)
            layer._w_offline_done = False
            quantizer.scale = base_scale * factor
            quantizer.zero = base_zero.clone()
            result = original_solver(layer, blocksize=blocksize, percdamp=percdamp)
            if not torch.isfinite(layer.weight).all():
                raise RuntimeError('Nonfinite quantized weights')
            score = output_error(layer.weight, reference, layer.H)
            if not torch.isfinite(torch.tensor(score)):
                raise RuntimeError('Nonfinite output reconstruction score')
            rows.append({'factor': factor, 'scale': (quantizer.scale.detach().cpu().flatten().tolist() if quantizer.perchannel
                                    else float(quantizer.scale.item())),
                         'calibration_output_error': score})
            if best is None or score < best['score']:
                best = {'score':score, 'factor':factor, 'weight':layer.weight.detach().clone(),
                        'scale':quantizer.scale.clone(), 'zero':quantizer.zero.clone(),
                        'diagnostics':result}
        layer.weight.data.copy_(best['weight'])
        quantizer.scale, quantizer.zero = best['scale'], best['zero']
        layer._w_offline_done = True
        return best['diagnostics'], {'selected_factor':best['factor'], 'candidates':rows,
                                    'granularity':'channel' if quantizer.perchannel else 'tensor',
                                    'factor_sharing':'one factor per layer, applied to all base scales',
                                    'baseline_error':rows[0]['calibration_output_error'],
                                    'selected_error':best['score']}
    except BaseException:
        layer.weight.data.copy_(reference)
        quantizer.scale, quantizer.zero, quantizer.maxq = old_scale, old_zero, old_maxq
        layer._w_offline_done = old_done
        raise


