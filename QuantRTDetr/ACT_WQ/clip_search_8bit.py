"""Shared, streaming 8-bit clipping search for channel and tensor calibration."""
import torch

FIELDS = ('clip', 'scale', 'zero', 'clip_min')
POLICY = 'expanded_8bit_v1'
NEG = (.995, .997, .999, .9995, .9997, .9999, .99995, .99999, 1.0)
POS = NEG
SINGLE = (.999, .9995, .9997, .9999, .99995, .99999, 1.0)


def choices(x):
    """Yield one error tensor at a time; preserve the historical 18 candidates first."""
    neg, pos = x[x < 0], x[x > 0]
    if not x.numel() or not torch.isfinite(x).all():
        raise ValueError('8-bit search requires finite, nonempty samples')
    if not len(neg) and not len(pos):
        raise ValueError('Keep the existing bootstrap for all-zero samples')

    def quantiles(values, levels):
        return {p: max((values.max() if p == 1 else torch.quantile(values, p)).item(), 1e-6)
                for p in levels}

    if len(neg) and len(pos):
        ns, ps = quantiles(neg.abs(), NEG), quantiles(pos, POS)
        old = [(n, p) for n in (.995, .997, .999)
               for p in (.995, .997, .999, .9995, .9997, .9999)]
        pairs = old + [(n, p) for n in NEG for p in POS if (n, p) not in old]
        bounds = [(-ns[n], ps[p], 'percentile:%g,%g' % (n, p)) for n, p in pairs]
    elif len(pos):
        ps = quantiles(pos, SINGLE)
        bounds = [(0., ps[p], 'positive:%g' % p) for p in SINGLE]
    else:
        ns = quantiles(neg.abs(), SINGLE)
        bounds = [(-ns[p], 0., 'negative:%g' % p) for p in SINGLE]

    # In addition to percentile=1, include an outward-aligned Min-Max grid.
    # Ordinary rounded-zero alignment may otherwise clip one endpoint.
    bounds.append((min(float(x.min()), 0.), max(float(x.max()), 0.), 'minmax_cover'))
    for lo, hi, label in bounds:
        scale = (hi - lo) / (255 + 1e-12)
        zero = int(max(0, min(255, round(-lo / scale))))
        if label == 'minmax_cover':
            if lo < 0 < hi:
                zero = max(1, min(254, zero))
            scale = max(-lo / zero if zero else 0., hi / (255-zero) if zero < 255 else 0., 1e-12)
        lo, hi = -zero * scale, (255-zero) * scale
        clipped = torch.clamp(x, lo, hi)
        q = torch.clamp(torch.round(clipped / scale + zero), 0, 255)
        error = ((q-zero) * scale - x).square()
        # Match the original float32 scoring and stored-parameter convention.
        stored_scale = (hi-lo) / (255 + 1e-12)
        stored_zero = int(max(0, min(255, round(-lo/stored_scale))))
        params = dict(zip(FIELDS, torch.tensor([hi, stored_scale, stored_zero, lo],
                                              dtype=torch.float32).tolist()))
        yield params, error, label


def select(x, weights=None):
    if weights is not None and (weights.shape != x.shape or not torch.isfinite(weights).all()
                                or (weights < 0).any()):
        raise ValueError('Invalid clipping weights')
    best_u = best_s = float('inf')
    u = s = None
    diag = {'candidate_policy': POLICY}
    for index, (params, error, label) in enumerate(choices(x)):
        mse = error.mean().item()
        weighted = mse if weights is None else (error * weights).mean().item()
        if mse < best_u:
            best_u, u = mse, params.copy()
            diag.update(selected_U=index, candidate_U=label, mse_U=mse, weighted_U=weighted)
        if weighted < best_s:
            best_s, s = weighted, params.copy()
            diag.update(selected_S=index, candidate_S=label, mse_S=mse, weighted_S=weighted)
    if u is None or s is None:
        raise RuntimeError('No finite 8-bit clipping candidate')
    diag['candidate_count'] = index + 1
    return u, s, diag
