import torch
import os
import math
import random
import numpy as np

# ------------------------------
# DEVICE
# ------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For reproducible CUDA algorithms (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ------------------------------
# SIMPLE HELPERS
# ------------------------------
def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct, targets.size(0)


# ------------------------------
# MODEL SIZE
# ------------------------------
def get_fp32_model_size_mb(model):
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * 4) / (1024 * 1024)

def get_pruned_model_size_mb(model):
    total_nnz = 0
    for p in model.parameters():
        total_nnz += torch.count_nonzero(p).item()
    return (total_nnz * 4) / (1024 * 1024)

def get_weight_quant_size_mb(model, bitwidth=8):
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * (bitwidth/8)) / (1024 * 1024)
import torch
import os


# -------------------------------------------------------
# 3) Generic weight quantized size
# -------------------------------------------------------
def get_weight_quant_size_mb(model, bitwidth):
    """
    Computes model size for INT2 / INT4 / INT6 / INT8.
    """
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = bitwidth / 8
    total_bytes = total_params * bytes_per_param
    return total_bytes / (1024 * 1024)


# -------------------------------------------------------
# 4) FP8 E5M3 model size
# -------------------------------------------------------
def get_fp8_e5m3_size_mb(model):
    """
    FP8 E5M3 uses 1 byte per parameter.
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = total_params * 1  # FP8 = 1 byte
    return total_bytes / (1024 * 1024)


# -------------------------------------------------------
# 5) FP8 E6M2 model size
# -------------------------------------------------------
def get_fp8_e6m2_size_mb(model):
    """
    FP8 E6M2 also uses 1 byte per parameter.
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = total_params * 1
    return total_bytes / (1024 * 1024)


def compression_ratio(original, compressed):
    return original / compressed


# ------------------------------
# FILE-BASED SIZE
# ------------------------------
def get_model_size_in_mb(model):
    torch.save(model.state_dict(), "tmp.pth")
    size = os.path.getsize("tmp.pth") / (1024 * 1024)
    os.remove("tmp.pth")
    return size


# ------------------------------
# MASK UTILITIES
# ------------------------------
def find_first_conv_and_last_linear(model):
    first_conv, last_linear = None, None
    for name, m in model.named_modules():
        if first_conv is None and isinstance(m, torch.nn.Conv2d):
            first_conv = name
        if isinstance(m, torch.nn.Linear):
            last_linear = name
    return first_conv, last_linear


def init_masks_from_model(model):
    masks = {}
    for name, param in model.named_parameters():
        if name.endswith(".weight"):
            masks[name] = torch.ones_like(param.data, dtype=torch.uint8, device='cpu')
    return masks


def apply_masks_to_model(model, masks):
    with torch.no_grad():
        for name, mask in masks.items():
            param = dict(model.named_parameters())[name]
            param.data.mul_(mask.to(param.device))


def compute_layer_sparsities(model, masks):
    out = []
    for name, mask in masks.items():
        n = mask.numel()
        nz = int(mask.sum().item())
        zero = n - nz
        out.append((name, n, zero, zero/n))
    return out


def overall_sparsity_from_masks(masks):
    total = sum(mask.numel() for mask in masks.values())
    zeros = sum(mask.numel() - int(mask.sum().item()) for mask in masks.values())
    return zeros / total

