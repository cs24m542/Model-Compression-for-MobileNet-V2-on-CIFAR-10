import torch
import torch.nn as nn
import copy
import numpy as np
from sklearn.cluster import KMeans

#############################################
# Utility: BatchNorm Recalibration
#############################################
def recalibrate_batchnorm(model, data_loader, device, num_batches=20):
    """
    Recompute BN running_mean / running_var after quantization.
    """
    model.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)
            model(images)
            if i >= num_batches:
                break
    model.eval()


#############################################
# K-MEANS WEIGHT QUANTIZATION
#############################################
def kmeans_quantize_tensor_channelwise(W, K=8):
    """
    Channel-wise K-means quantization for Conv/Linear tensors.
    """
    W_cpu = W.detach().cpu().numpy()
    Wq = np.zeros_like(W_cpu, dtype=np.float32)
    out_ch = W_cpu.shape[0]

    for c in range(out_ch):
        w_flat = W_cpu[c].reshape(-1, 1)
        kmeans = KMeans(n_clusters=K, n_init=5, random_state=0).fit(w_flat)
        centroids = kmeans.cluster_centers_.squeeze()
        labels = kmeans.labels_
        Wq[c] = centroids[labels].reshape(W_cpu[c].shape)

    return torch.tensor(Wq, dtype=W.dtype, device=W.device)


def kmeans_quantize_model(model, K=8):
    """
    Applies channel-wise K-means quantization to all Conv/Linear weights.
    """
    qmodel = copy.deepcopy(model)
    for name, param in qmodel.named_parameters():
        if "weight" not in name:
            continue
        if param.dim() >= 2:
            q_weight = kmeans_quantize_tensor_channelwise(param.data, K=K)
            param.data = q_weight
    return qmodel


#############################################
# INT-N WEIGHT QUANTIZATION (2,4,6,8 bits)
#############################################
def quantize_weight_intN(w: torch.Tensor, n_bits: int):
    assert n_bits in [2, 4, 6, 8], f"Unsupported bit-width: {n_bits}"
    max_int = 2 ** (n_bits - 1) - 1

    max_val = w.abs().max()
    if max_val == 0:
        return w.clone()

    scale = max_val / max_int
    w_int = torch.clamp((w / scale).round(), -max_int - 1, max_int)
    return w_int * scale


def quantize_int8_tensor(W):
    W_fp = W.detach()
    max_val = W_fp.abs().max()
    scale = max_val / 127
    W_int = torch.clamp((W_fp / scale).round(), -127, 127).to(torch.int8)
    return (W_int.float() * scale).to(W.dtype)


#############################################
# FP8 Weight Quantization
#############################################
def quantize_fp8_e5m3_tensor(W):
    scale = W.abs().max() / 448
    W_scaled = W / scale
    W_fp8 = torch.round(W_scaled * 8) / 8
    return (W_fp8 * scale).to(W.dtype)


def quantize_fp8_e6m2_tensor(W):
    scale = W.abs().max() / 57344
    W_scaled = W / scale
    W_fp8 = torch.round(W_scaled * 4) / 4
    return (W_fp8 * scale).to(W.dtype)


#############################################
# BFLOAT16
#############################################
def quantize_bfloat16(model):
    qmodel = copy.deepcopy(model)
    return qmodel.to(torch.bfloat16)


#############################################
# APPLY WEIGHT QUANTIZATION (Main Wrapper)
#############################################
def apply_weight_quantization(model,
                              method="kmeans",
                              K=8,
                              train_loader=None,
                              device=None):
    """
    Applies any of the supported quantization methods to a model.
    Returns the quantized model.
    """
    qmodel = copy.deepcopy(model)

    if method == "kmeans":
        qmodel = kmeans_quantize_model(qmodel, K)
        if train_loader is not None:
            recalibrate_batchnorm(qmodel, train_loader, device, num_batches=50)

    else:
        for name, param in qmodel.named_parameters():
            if "weight" not in name:
                continue

            if method == "int8":
                param.data = quantize_int8_tensor(param.data)

            elif method == "int2":
                param.data = quantize_weight_intN(param.data, 2)

            elif method == "int4":
                param.data = quantize_weight_intN(param.data, 4)

            elif method == "int6":
                param.data = quantize_weight_intN(param.data, 6)

            elif method == "fp8_e5m3":
                param.data = quantize_fp8_e5m3_tensor(param.data)

            elif method == "fp8_e6m2":
                param.data = quantize_fp8_e6m2_tensor(param.data)

            elif method == "bfloat16":
                param.data = param.data.to(torch.bfloat16)

            else:
                raise ValueError(f"Unknown weight quantization: {method}")

    return qmodel

