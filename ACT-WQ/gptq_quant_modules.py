# gptq_quant_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import copy
from typing import Optional

def quantize(x, scale, zero, maxq):
    if isinstance(maxq, torch.Tensor):
        maxq_val = int(maxq.item())
    else:
        maxq_val = int(maxq)
    if maxq_val < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq_val)
    return scale * (q - zero)

# gptq_quant_modules.py: ActQuantizer 增加 clip_min 支持
class ActQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('maxq', torch.tensor(0, dtype=torch.int32))
        self.register_buffer('scale', torch.tensor(0.0))
        self.register_buffer('zero',  torch.tensor(0.0))
        self.register_buffer('clip',  torch.tensor(0.0))
        self.register_buffer('clip_min', torch.tensor(0.0))  # 新增：下限
        self._enabled = False
        self._ready = False
        self._symmetric = True
        self._perchannel = False
        self._ch_axis = 1

    def configure(self, bits: int, symmetric: bool = True, perchannel: bool = False, ch_axis: int = 1):
        self.maxq = torch.tensor(2**bits - 1, dtype=torch.int32, device=self.maxq.device)
        self._symmetric = symmetric
        self._perchannel = perchannel
        self._ch_axis = ch_axis
        self._enabled = True
        self._ready = False

    @torch.no_grad()
    def set_params(self, scale, zero, clip, clip_min=0.0):  # 支持 clip_min
        def to_1d_tensor(v, like):
            if isinstance(v, torch.Tensor):
                return v.to(dtype=torch.float32, device=like.device).detach().clone()
            elif isinstance(v, (list, tuple)):
                return torch.tensor(v, dtype=torch.float32, device=like.device)
            else:
                return torch.tensor([float(v)], dtype=torch.float32, device=like.device)
        self.scale    = to_1d_tensor(scale, self.maxq)
        self.zero     = to_1d_tensor(zero,  self.maxq)
        self.clip     = to_1d_tensor(clip,  self.maxq)
        self.clip_min = to_1d_tensor(clip_min, self.maxq)
        self._ready = True

    def _broadcast_params(self, x: torch.Tensor):
        if not self._perchannel:
            return self.scale.view(1), self.zero.view(1), self.clip.view(1), self.clip_min.view(1)
        ch_axis = self._ch_axis if self._ch_axis >= 0 else (x.dim() + self._ch_axis)
        C = x.size(ch_axis)
        shape = [1] * x.dim(); shape[ch_axis] = C
        return (
            self.scale.view(shape),
            self.zero.view(shape),
            self.clip.view(shape),
            self.clip_min.view(shape),
        )

    def ready(self):
        return self._ready and torch.all(self.scale != 0)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.ready():
            return x
        s, z, cmax, cmin = self._broadcast_params(x)
        # clamp 到 [clip_min, clip]
        x = torch.maximum(torch.minimum(x, cmax), cmin)
        return quantize(x, s, z, self.maxq)

# ---------------- Weight quantizer (unchanged) ----------------
class GPTQQuantizer(nn.Module):
    def __init__(self, shape=1):
        super(GPTQQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        self.perchannel = False
        self.sym = True
        self.mse = False
        self.norm = 2.4
        self.grid = 100
        self.maxshrink = 0.8

    def configure(self, bits, perchannel=False, sym=True, mse=False,
                  norm=2.4, grid=100, maxshrink=0.8, trits=False):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)
        original_shape = x.shape
        if self.perchannel:
            if weight:
                if len(original_shape) == 4:
                    x = x.reshape(original_shape[0], -1)
                elif len(original_shape) == 2:
                    x = x
            else:
                if len(original_shape) == 4:
                    x = x.permute([1, 0, 2, 3]).flatten(1)
                elif len(original_shape) == 3:
                    x = x.reshape((-1, original_shape[-1])).t()
                elif len(original_shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.perchannel:
            if weight:
                pass
            else:
                if len(original_shape) == 4:
                    tmpn = original_shape[1]
                elif len(original_shape) == 3:
                    tmpn = original_shape[2]
                elif len(original_shape) == 2:
                    tmpn = original_shape[1]
                else:
                    tmpn = 1
                self.scale = self.scale.repeat(tmpn)
                self.zero = self.zero.repeat(tmpn)

        if weight:
            if self.perchannel:
                if len(original_shape) == 4:
                    self.scale = self.scale.reshape(-1, 1, 1, 1)
                    self.zero  = self.zero.reshape(-1, 1, 1, 1)
                elif len(original_shape) == 2:
                    self.scale = self.scale.reshape(-1, 1)
                    self.zero  = self.zero.reshape(-1, 1)
            else:
                self.scale = self.scale.reshape(1)
                self.zero  = self.zero.reshape(1)
            return

        if len(original_shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero  = self.zero.reshape((1, -1, 1, 1))
        elif len(original_shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero  = self.zero.reshape((1, 1, -1))
        elif len(original_shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero  = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

# ---- GPTQConv2d / GPTQLinear: 激活接口改造 ----

class GPTQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, bits=8, perchannel=True, sym=True):
        super(GPTQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.w_quantizer = GPTQQuantizer()
        self.w_quantizer.configure(bits=bits, perchannel=perchannel, sym=sym)
        self.a_quantizer = ActQuantizer()
        self.H = None; self.nsamples = 0; self.device = None
        self._w_offline_done = False

    # 新：支持 per-channel 激活量化
    def enable_activation_quant(self, bits: int = 8, symmetric: bool = True,
                                perchannel: bool = True, ch_axis: int = 1):
        self.a_quantizer.configure(bits=bits, symmetric=symmetric, perchannel=perchannel, ch_axis=ch_axis)

    @torch.no_grad()
    def set_activation_qparams(self, scale, zero, clip, clip_min=0.0):
        self.a_quantizer.set_params(scale, zero, clip, clip_min)

    # ---- weight GPTQ data collection ----
    def add_batch(self, inp, out):
        if self.device is None:
            self.device = inp.device
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        unfold = nn.Unfold(
            self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )
        inp = unfold(inp)
        inp = inp.permute([1, 0, 2]).flatten(1)
        inp = inp - inp.mean(dim=1, keepdim=True)  # DC centering
        if self.H is None:
            self.H = torch.zeros((inp.shape[0], inp.shape[0]), device=self.device)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def gptq_quantize(self, blocksize=128, percdamp=0.01):
        W = self.weight.data.clone()
        original_shape = W.shape
        W = W.flatten(1)
        W = W.float()
        diagnostics = {'cond_before_damp': float('inf'), 'cond_after_damp': float('inf')}
        if not self.w_quantizer.ready():
            self.w_quantizer.find_params(W, weight=True)
        if self.H is None:
            self.w_quantizer.find_params(W, weight=True)
            return diagnostics
        H = self.H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        try:
            diagnostics['cond_before_damp'] = float(torch.linalg.cond(H))
        except torch.linalg.LinAlgError:
            pass
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=self.device)
        H[diag, diag] += damp
        try:
            diagnostics['cond_after_damp'] = float(torch.linalg.cond(H))
        except torch.linalg.LinAlgError:
            diagnostics['cond_after_damp'] = float('inf')

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        Q = torch.zeros_like(W)
        for i1 in range(0, W.shape[1], blocksize):
            i2 = min(i1 + blocksize, W.shape[1])
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = quantize(w.unsqueeze(1), self.w_quantizer.scale, self.w_quantizer.zero, self.w_quantizer.maxq).flatten()
                Q1[:, i] = q
                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Q[:, i1:i2] = Q1
            W[:, i2:] -= (W1 - Q1).matmul(Hinv[i1:i2, i2:])
        self.weight.data = Q.reshape(original_shape).to(self.weight.data.dtype)
        self._w_offline_done = True
        return diagnostics

    def forward(self, x):
        # activation quant first
        if self.a_quantizer.ready():
            x = self.a_quantizer.quantize(x)
        # weight quant
        if (not self._w_offline_done) and self.w_quantizer.ready():
            original_shape = self.weight.shape
            w_flat = self.weight.flatten(1)
            w_quantized_flat = self.w_quantizer.quantize(w_flat)
            w = w_quantized_flat.reshape(original_shape)
        else:
            w = self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

# ---------------- GPTQLinear with activation quant ----------------
class GPTQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bits=8, perchannel=True, sym=True):
        super(GPTQLinear, self).__init__(in_features, out_features, bias)
        self.w_quantizer = GPTQQuantizer()
        self.w_quantizer.configure(bits=bits, perchannel=perchannel, sym=sym)
        self.a_quantizer = ActQuantizer()
        self.H = None; self.nsamples = 0; self.device = None
        self._w_offline_done = False

    def enable_activation_quant(self, bits: int = 8, symmetric: bool = True,
                                perchannel: bool = True, ch_axis: int = -1):
        self.a_quantizer.configure(bits=bits, symmetric=symmetric, perchannel=perchannel, ch_axis=ch_axis)

    @torch.no_grad()
    def set_activation_qparams(self, scale, zero, clip, clip_min=0.0):
        self.a_quantizer.set_params(scale, zero, clip, clip_min)

    def add_batch(self, inp, out):
        if self.device is None:
            self.device = inp.device
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t()
        inp = inp - inp.mean(dim=1, keepdim=True)  # DC centering
        if self.H is None:
            self.H = torch.zeros((inp.shape[0], inp.shape[0]), device=self.device)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def gptq_quantize(self, blocksize=128, percdamp=0.01):
        W = self.weight.data.clone().float()
        diagnostics = {'cond_before_damp': float('inf'), 'cond_after_damp': float('inf')}
        if not self.w_quantizer.ready():
            self.w_quantizer.find_params(W, weight=True)
        if self.H is None:
            self.w_quantizer.find_params(W, weight=True)
            return diagnostics
        H = self.H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        try:
            diagnostics['cond_before_damp'] = float(torch.linalg.cond(H))
        except torch.linalg.LinAlgError:
            pass
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=self.device)
        H[diag, diag] += damp
        try:
            diagnostics['cond_after_damp'] = float(torch.linalg.cond(H))
        except torch.linalg.LinAlgError:
            diagnostics['cond_after_damp'] = float('inf')

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        Q = torch.zeros_like(W)
        for i1 in range(0, W.shape[1], blocksize):
            i2 = min(i1 + blocksize, W.shape[1])
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = quantize(w.unsqueeze(1), self.w_quantizer.scale, self.w_quantizer.zero, self.w_quantizer.maxq).flatten()
                Q1[:, i] = q
                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Q[:, i1:i2] = Q1
            W[:, i2:] -= (W1 - Q1).matmul(Hinv[i1:i2, i2:])
        self.weight.data = Q.to(self.weight.data.dtype)
        self._w_offline_done = True
        return diagnostics

    def forward(self, x):
        if self.a_quantizer.ready():
            x = self.a_quantizer.quantize(x)
        if (not self._w_offline_done) and self.w_quantizer.ready():
            w = self.w_quantizer.quantize(self.weight)
        else:
            w = self.weight
        return F.linear(x, w, self.bias)
