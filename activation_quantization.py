import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans

############################################################
# GLOBAL LOGGERS & FLAGS
############################################################
ACTIVATION_LOG = {}              # Stores original & quantized tensors
ACTIVATION_HOOK_HANDLES = []     # Track forward hooks
ACT_QUANT_ENABLED = True         # Allows disabling quantization
disable_activation_logging = False


def enable_activation_quant():
    global ACT_QUANT_ENABLED
    ACT_QUANT_ENABLED = True

def disable_activation_quant():
    global ACT_QUANT_ENABLED
    ACT_QUANT_ENABLED = False



############################################################
# GPU-SAFE APPROXIMATE PERCENTILE
############################################################
def approx_percentile_gpu(x, p=0.999, bins=2048):
    """
    Memory-safe percentile approximation using histogram.
    Works on GPU, avoids torch.quantile OOM.
    """
    x = x.detach()
    x_min = x.min()
    x_max = x.max()

    hist = torch.histc(x, bins=bins, min=x_min.item(), max=x_max.item())
    cdf = torch.cumsum(hist, dim=0)
    target = cdf[-1] * p

    idx = torch.searchsorted(cdf, target)
    bin_width = (x_max - x_min) / bins

    return x_min + idx * bin_width


############################################################
# K-MEANS ACTIVATION QUANTIZATION (fast uniform bins)
############################################################
def quantize_act_kmeans_fast(x, K=8):
    """
    Fast uniform-binning approximation of K-means.
    """
    xmin = x.min()
    xmax = x.max()

    if (xmax - xmin) < 1e-8:
        return x.clone()

    centers = torch.linspace(xmin, xmax, K, device=x.device)
    step = (xmax - xmin) / (K - 1)

    idx = torch.clamp(((x - xmin) / step).round(), 0, K - 1).long()
    return centers[idx]


############################################################
# INT8 ACTIVATION QUANTIZATION
############################################################
def quantize_act_int8(x, layer_name):
    """
    INT8 activation quantization with percentile clipping.
    Per-tensor (not per-channel) to avoid OOM.
    """
    lo = approx_percentile_gpu(x, p=0.001)
    hi = approx_percentile_gpu(x, p=0.999)

    x_clipped = x.clamp(lo, hi)
    scale = 255.0 / (hi - lo + 1e-8)

    x_int = torch.round((x_clipped - lo) * scale).clamp(0, 255)
    x_quant = (x_int / scale) + lo

    # Store logs
    if not disable_activation_logging:
        if layer_name not in ACTIVATION_LOG:
            ACTIVATION_LOG[layer_name] = {}
        ACTIVATION_LOG[layer_name]["original"] = x.detach().float().cpu()
        ACTIVATION_LOG[layer_name]["quantized"] = x_quant.detach().float().cpu()

    return x_quant


############################################################
# FP8 E5M3 ACTIVATION QUANTIZATION
############################################################
def quantize_act_fp8_e5m3(x, layer_name="unknown", bins=512):
    C = x.shape[1]
    x_reshaped = x.permute(1,0,2,3).contiguous().view(C, -1)

    q = torch.zeros_like(x)
    step = 1 / 8  # 3 mantissa bits

    for c in range(C):
        xc = x_reshaped[c]

        lo = approx_percentile_gpu(xc, p=0.001, bins=bins)
        hi = approx_percentile_gpu(xc, p=0.999, bins=bins)
        xc = xc.clamp(lo, hi)

        max_val = max(abs(lo.item()), abs(hi.item()))
        scale = max_val / 240.0  # FP8 E5M3 max integer

        if scale < 1e-8:
            q_channel = xc
        else:
            xn = xc / scale
            xn_q = torch.round(xn / step) * step
            xn_q = xn_q.clamp(-240, 240)
            q_channel = xn_q * scale

        q[:, c] = q_channel.view_as(q[:, c])

    if not disable_activation_logging:
        ACTIVATION_LOG[layer_name] = {
            "original": x.detach().cpu(),
            "quantized": q.detach().cpu(),
            "quant_bits": 8
        }

    return q


############################################################
# FP8 E6M2 ACTIVATION QUANTIZATION
############################################################
def quantize_act_fp8_e6m2(x, layer_name="unknown", bins=512):
    C = x.shape[1]
    x_reshaped = x.permute(1,0,2,3).contiguous().view(C, -1)
    q = torch.zeros_like(x)
    step = 1 / 4  # 2 mantissa bits

    for c in range(C):
        xc = x_reshaped[c]

        lo = approx_percentile_gpu(xc, p=0.001, bins=bins)
        hi = approx_percentile_gpu(xc, p=0.999, bins=bins)
        xc = xc.clamp(lo, hi)

        max_val = max(abs(lo.item()), abs(hi.item()))
        scale = max_val / 448.0  # FP8 E6M2 max integer

        if scale < 1e-8:
            q_channel = xc
        else:
            xn = xc / scale
            xn_q = torch.round(xn / step) * step
            xn_q = xn_q.clamp(-448, 448)
            q_channel = xn_q * scale

        q[:, c] = q_channel.view_as(q[:, c])

    if not disable_activation_logging:
        ACTIVATION_LOG[layer_name] = {
            "original": x.detach().cpu(),
            "quantized": q.detach().cpu(),
            "quant_bits": 8
        }

    return q


############################################################
# HOOK MANAGEMENT — ADDING QUANTIZATION HOOKS
############################################################
def apply_activation_quantization(model, method):
    """
    Registers forward hooks for activation quantization.
    Supported: int8, fp8_e5m3, fp8_e6m2
    """
    remove_activation_quant_hooks()

    global ACT_QUANT_ENABLED
    ACT_QUANT_ENABLED = True

    if method == "int8":
        qfunc = quantize_act_int8
    elif method == "fp8_e5m3":
        qfunc = quantize_act_fp8_e5m3
    elif method == "fp8_e6m2":
        qfunc = quantize_act_fp8_e6m2
    else:
        raise ValueError(f"Unknown activation quantization: {method}")

    for name, module in model.named_modules():

        if isinstance(module, (nn.ReLU, nn.ReLU6, nn.Conv2d, nn.Linear)):

            def make_hook(qfunc, lname):
                def hook_fn(m, inp, out):
                    if not ACT_QUANT_ENABLED:
                        return out
                    return qf(out, lname)
                return hook_fn

            handle = module.register_forward_hook(make_hook(qfunc, name))
            ACTIVATION_HOOK_HANDLES.append(handle)

    return ACTIVATION_HOOK_HANDLES


############################################################
# REMOVE HOOKS
############################################################
def remove_activation_quant_hooks():
    global ACTIVATION_HOOK_HANDLES

    if len(ACTIVATION_HOOK_HANDLES) == 0:
        return

    for h in ACTIVATION_HOOK_HANDLES:
        try:
            h.remove()
        except:
            pass

    ACTIVATION_HOOK_HANDLES = []


############################################################
# COMPUTE ACTIVATION COMPRESSION
############################################################
def compute_activation_compression():
    """
    Computes MB saved and compression ratio for each layer.
    """
    records = []

    for layer_name, actdict in ACTIVATION_LOG.items():

        orig = actdict["original"]
        quant = actdict["quantized"]

        orig_bits = orig.numel() * 32
        quant_bits = quant.numel() * 8

        records.append({
            "layer": layer_name,
            "orig_MB": orig_bits / (8 * 1024 * 1024),
            "quant_MB": quant_bits / (8 * 1024 * 1024),
            "compression_ratio": orig_bits / quant_bits
        })

    import pandas as pd
    return pd.DataFrame(records)


############################################################
# WEIGHTED COMPRESSION RATIO
############################################################
def compute_weighted_act_ratio(df):
    """
    Weighted average of activation compression ratio.
    Weights = activation original size (MB)
    """
    weights = df["orig_MB"]
    ratios = df["compression_ratio"]
    return (weights * ratios).sum() / weights.sum()

