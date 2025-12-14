"""
compression.py
PART 1 — Core Pruning Utilities
This file assumes a single import block at top (Option 1).
"""

import torch
import torch.nn as nn
import copy
import math
from typing import Dict, List, Tuple
import time
import pandas as pd
from evaluation import evaluate
from training import recalibrate_batchnorm
from activation_quantization import (
    apply_activation_quantization,
    remove_activation_quant_hooks
)
import activation_quantization as aq

from weight_quantization import (
    apply_weight_quantization
)

from utility import (
    get_fp32_model_size_mb,
    get_pruned_model_size_mb,
    get_weight_quant_size_mb,
    get_fp8_e5m3_size_mb,
    get_fp8_e6m2_size_mb
)

def find_first_conv_and_last_linear(model: nn.Module):
    """
    Identifies first Conv2d and last Linear layer.
    Needed for protecting early/late layers from over-pruning.
    """
    first_conv = None
    last_linear = None

    for name, m in model.named_modules():
        if first_conv is None and isinstance(m, nn.Conv2d):
            first_conv = name
        if isinstance(m, nn.Linear):
            last_linear = name

    return first_conv, last_linear

def init_masks_from_model(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Creates binary masks (all ones) for every Conv/Linear weight tensor.
    Masks stored on CPU as uint8.
    """
    masks = {}
    for name, param in model.named_parameters():
        if name.endswith(".weight"):
            masks[name] = torch.ones_like(param.data, dtype=torch.uint8, device='cpu')
    return masks

def apply_masks_to_model(model: nn.Module, masks: Dict[str, torch.Tensor]):
    """
    Applies masks to model parameters in-place.
    Ensures pruned weights stay zero during training.
    """
    with torch.no_grad():
        for name, mask in masks.items():
            try:
                param = dict(model.named_parameters())[name]
            except KeyError:
                continue
            param.data.mul_(mask.to(param.device))


def compute_layer_sparsities(model: nn.Module, masks: Dict[str, torch.Tensor]) -> List[Tuple[str,int,int,float]]:
    """
    Returns statistics per layer:
    (param_name, #params, #zeros, sparsity_fraction)
    """
    stats = []
    for name, mask in masks.items():
        total = mask.numel()
        nz = int(mask.sum().item())
        zeros = total - nz
        sparsity = zeros / total
        stats.append((name, total, zeros, sparsity))
    return stats


def overall_sparsity_from_masks(masks: Dict[str, torch.Tensor]) -> float:
    total = 0
    zeros = 0
    for mask in masks.values():
        total += mask.numel()
        zeros += (mask.numel() - int(mask.sum().item()))
    return zeros / total if total > 0 else 0.0


def compute_l1_layer_scores(model: nn.Module, masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Computes L1 norm of masked weights, used for sensitivity-based allocation.
    """
    scores = {}
    params = dict(model.named_parameters())

    for name, mask in masks.items():
        w = params[name]
        m = mask.to(w.device)
        score = (w.data.abs() * m).sum().item()
        scores[name] = max(score, 1e-12)  # avoid zeros

    return scores

def allocate_nonzeros_by_scores(
    masks: Dict[str, torch.Tensor],
    scores: Dict[str, float],
    global_sparsity: float,
    gamma: float,
    s_min: float,
    s_max: float,
    first_layer_names: List[str],
    last_layer_names: List[str],
    first_layer_cap: float,
    last_layer_cap: float
) -> Dict[str, int]:
    """
    Distributes nonzero weights to each layer based on sensitivity scores.
    Sensitivity^gamma controls aggressiveness.
    Includes special caps for first conv and last linear.
    """

    # Total parameters
    param_counts = {name: mask.numel() for name, mask in masks.items()}
    total_params = sum(param_counts.values())

    desired_nnz_total = int(round((1 - global_sparsity) * total_params))

    # Sensitivity power transform
    a = {name: scores[name] ** gamma for name in masks}
    A = sum(a.values())

    # Raw proportional allocation
    alloc = {}
    for name in masks:
        frac = a[name] / A if A > 0 else 0.0
        alloc[name] = max(1, int(round(frac * desired_nnz_total)))

    # Enforce bounds per layer
    for name in alloc:
        n = param_counts[name]
        max_nnz = int((1 - s_min) * n)
        min_nnz = int(max(1, (1 - s_max) * n))

        # Special case: first conv
        if name in first_layer_names:
            min_nnz = max(min_nnz, int((1 - first_layer_cap) * n))

        # Special case: last linear
        if name in last_layer_names:
            min_nnz = max(min_nnz, int((1 - last_layer_cap) * n))

        alloc[name] = min(max(alloc[name], min_nnz), max_nnz)

    # Adjust allocations to match exactly desired total nonzeros
    diff = desired_nnz_total - sum(alloc.values())

    # Need to add nonzeros
    if diff > 0:
        ordered = sorted(alloc, key=lambda x: scores[x], reverse=True)
        i = 0
        while diff > 0:
            layer = ordered[i % len(ordered)]
            n = param_counts[layer]
            max_nnz = int((1 - s_min) * n)
            if alloc[layer] < max_nnz:
                alloc[layer] += 1
                diff -= 1
            i += 1

    # Need to remove nonzeros
    if diff < 0:
        diff = -diff
        ordered = sorted(alloc, key=lambda x: scores[x])
        i = 0
        while diff > 0:
            layer = ordered[i % len(ordered)]
            n = param_counts[layer]
            min_nnz = int(max(1, (1 - s_max) * n))

            # special protections
            if layer in first_layer_names:
                min_nnz = max(min_nnz, int((1 - first_layer_cap) * n))
            if layer in last_layer_names:
                min_nnz = max(min_nnz, int((1 - last_layer_cap) * n))

            if alloc[layer] > min_nnz:
                alloc[layer] -= 1
                diff -= 1
            i += 1

    return alloc


def update_masks_by_alloc(model: nn.Module, masks: Dict[str, torch.Tensor], alloc_nonzeros: Dict[str, int]):
    """
    For each parameter: keep only alloc_nonzeros[name] largest |w| among currently unmasked entries.
    Updates masks in-place.
    """

    params = dict(model.named_parameters())

    for name, desired_nz in alloc_nonzeros.items():

        if name not in masks:
            continue

        mask = masks[name]
        w = params[name].data.detach().cpu()
        m = mask.to(torch.bool)

        current_nz = int(m.sum().item())

        # No pruning needed
        if desired_nz >= current_nz:
            continue

        # Only consider active weights
        flat_w = w.view(-1)
        flat_m = m.view(-1)
        active_idx = torch.nonzero(flat_m).view(-1)

        if active_idx.numel() == 0:
            continue

        active_vals = flat_w[active_idx].abs()

        num_prune = current_nz - desired_nz

        # prune smallest magnitudes
        _, prune_idx_local = torch.topk(active_vals, num_prune, largest=False)
        prune_idx = active_idx[prune_idx_local]

        flat_m[prune_idx] = False

        masks[name] = flat_m.view_as(mask).to(torch.uint8)

    return masks



def finetune_with_masks(
        model: nn.Module,
        masks: Dict[str, torch.Tensor],
        train_loader,
        val_loader,
        device,
        epochs=2,
        lr=1e-3,
        weight_decay=5e-4,
        checkpoint_prefix=None):
    """
    Fine-tunes a masked/pruned model while enforcing masks after every update.
    Returns best validation accuracy and corresponding state_dict.
    """

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    best_val_acc = -1.0
    best_state = None

    named_params = dict(model.named_parameters())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # enforce mask on gradients
            for pname, p in named_params.items():
                if pname in masks and p.grad is not None:
                    p.grad.data.mul_(masks[pname].to(p.grad.device))

            optimizer.step()

            # enforce mask on weights
            with torch.no_grad():
                for pname, p in named_params.items():
                    if pname in masks:
                        p.data.mul_(masks[pname].to(p.device))

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # validation
        val_loss, val_acc = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)

        print(f"[Finetune] Epoch {epoch}/{epochs} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_prefix:
                torch.save({
                    "model_state_dict": best_state,
                    "val_acc": best_val_acc
                }, f"{checkpoint_prefix}_best.pth")

    return best_val_acc, best_state

def iterative_layerwise_prune_and_finetune(
    base_model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device,
    target_sparsities: List[float] = [0.3, 0.5, 0.7],
    K_steps: int = 10,
    finetune_epochs_per_step: int = 2,
    gamma: float = 1.0,
    s_min: float = 0.0,
    s_max: float = 0.95,
    exclude_first_last: bool = True,
    first_last_cap: float = 0.2,
    stop_if_within_delta: float = 1.0,
    baseline_val_acc: float = None,
    output_csv: str = "prune_sweep_results.csv",
    checkpoint_dir: str = "./prune_checkpoints"):
    """
    Performs iterative pruning for each target sparsity in target_sparsities.
    Returns DataFrame of results and final pruned model.
    """

    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    results = []

    # detect first conv & last linear
    first_conv, last_linear = find_first_conv_and_last_linear(base_model)

    first_layer_names = []
    last_layer_names = []

    for pname, _ in base_model.named_parameters():
        if first_conv and pname.startswith(first_conv):
            first_layer_names.append(pname)
        if last_linear and pname.startswith(last_linear):
            last_layer_names.append(pname)

    print("First Layer Params (10% max sparsity):")
    for n in first_layer_names:
        print("   ", n)

    print("Last Layer Params (20% max sparsity):")
    for n in last_layer_names:
        print("   ", n)

    # Baseline accuracy if not provided
    if baseline_val_acc is None:
        baseline_val_acc = evaluate(base_model, val_loader, nn.CrossEntropyLoss(), device)[1]
        print("Computed baseline val acc:", baseline_val_acc)

    # Loop over target sparsities
    for target_s in target_sparsities:
        print("\n" + "="*60)
        print(f"Beginning pruning for target sparsity = {target_s:.3f}")

        model = copy.deepcopy(base_model).to(device)
        masks = init_masks_from_model(model)
        apply_masks_to_model(model, masks)

        best_val_for_target = -1.0
        best_state_for_target = None
        start_time = time.time()

        # iterative pruning schedule
        for step in range(1, K_steps + 1):
            frac = step / K_steps
            current_global_s = target_s * (1 - (1 - frac)**3)

            print(f"\n-- Step {step}/{K_steps}: target_sparsity = {current_global_s:.4f}")

            # sensitivity
            scores = compute_l1_layer_scores(model, masks)

            # allocate nonzeros
            alloc = allocate_nonzeros_by_scores(
                masks=masks,
                scores=scores,
                global_sparsity=current_global_s,
                gamma=gamma,
                s_min=s_min,
                s_max=s_max,
                first_layer_names=first_layer_names,
                last_layer_names=last_layer_names,
                first_layer_cap=0.10,
                last_layer_cap=0.20
            )

            # update masks
            masks = update_masks_by_alloc(model, masks, alloc)
            apply_masks_to_model(model, masks)

            cp_prefix = f"{checkpoint_dir}/pruned_s{int(100*target_s)}_step{step}"

            val_after_ft, state_after_ft = finetune_with_masks(
                model,
                masks,
                train_loader,
                val_loader,
                device,
                epochs=finetune_epochs_per_step,
                lr=1e-3,
                weight_decay=5e-4,
                checkpoint_prefix=cp_prefix
            )

            best_state_for_target = copy.deepcopy(state_after_ft)

            current_sparsity = overall_sparsity_from_masks(masks)
            print(f"After step {step}: sparsity={current_sparsity:.4f} | val_acc={val_after_ft:.2f}%")

            # Early stopping
            if (abs(current_sparsity - target_s) <= 0.01) and \
               (val_after_ft >= baseline_val_acc - stop_if_within_delta):
                print("Early stop: target reached with stable accuracy.")
                break

        elapsed = time.time() - start_time

        # load best weights
        if best_state_for_target is not None:
            model.load_state_dict(best_state_for_target)

        apply_masks_to_model(model, masks)

        # final eval
        train_loss, train_acc = evaluate(model, train_loader, nn.CrossEntropyLoss(), device)
        val_loss, val_acc = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
        test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

        row = {
            "target_sparsity": target_s,
            "achieved_sparsity": overall_sparsity_from_masks(masks),
            "baseline_val_acc": baseline_val_acc,
            "final_val_acc": val_acc,
            "final_test_acc": test_acc,
            "final_train_acc": train_acc,
            "elapsed_seconds": elapsed,
            "K_steps": K_steps,
            "finetune_epochs_per_step": finetune_epochs_per_step,
            "gamma": gamma
        }
        results.append(row)

        # save checkpoint
        ckpt = f"{checkpoint_dir}/pruned_target_s{int(100*target_s)}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "masks": {k: v.clone() for k, v in masks.items()},
            "summary": row
        }, ckpt)
        print(f"Saved pruned model to: {ckpt}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved pruning summary to {output_csv}")

    return df, model



def Do_pruning(model_loaded,
               train_loader,
               val_loader,
               test_loader,
               device,
               target_sparsity,
               K_steps,
               finetune_epochs_per_step,
               baseline_val_acc):

    df_results, pruned_model = iterative_layerwise_prune_and_finetune(
        base_model=copy.deepcopy(model_loaded),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        target_sparsities=[target_sparsity],
        K_steps=K_steps,
        finetune_epochs_per_step=finetune_epochs_per_step,
        gamma=1.0,
        s_min=0.0,
        s_max=0.95,
        exclude_first_last=True,
        first_last_cap=0.2,
        stop_if_within_delta=1.0,
        baseline_val_acc=baseline_val_acc,
        output_csv="prune_sweep_results.csv",
        checkpoint_dir="./prune_checkpoints"
    )

    return df_results, pruned_model



def full_compression_pipeline(
    orig_model,
    pruned_model,
    train_loader,
    val_loader,
    test_loader,
    device,
    weight_method="int8",
    act_method="int8",
    K=8
):
    """
    Full pipeline:
        1) Start from PRUNED model
        2) Apply WEIGHT quantization
        3) Apply ACTIVATION quantization (hook-based)
        4) Evaluate
        5) Compute compression statistics
    """

    # Make a working copy
    work_model = copy.deepcopy(pruned_model).to(device)

    # ----------------------------------------------
    # STEP 1 — WEIGHT QUANTIZATION
    # ----------------------------------------------
    print("\n====================")
    print(f"STEP 1: Weight Quantization ({weight_method})")
    print("====================")

    wq_model = apply_weight_quantization(
        work_model,
        method=weight_method,
        K=K
    ).to(device)

    # Optional BN recalibration
    # recalibrate_batchnorm(wq_model, train_loader, device)

    _, wq_val_acc = evaluate(wq_model, val_loader, nn.CrossEntropyLoss(), device)
    print(f"Weight-Quantized Val Acc = {wq_val_acc:.2f}%")

    # ----------------------------------------------
    # STEP 2 — ACTIVATION QUANTIZATION
    # ----------------------------------------------
    print("\n====================")
    print(f"STEP 2: Activation Quantization ({act_method})")
    print("====================")

    aq_model = copy.deepcopy(wq_model).to(device)

    # Register activation hooks
    aq.enable_activation_quant()
    hooks = apply_activation_quantization(aq_model, method=act_method)

    remove_activation_quant_hooks()
    aq.disable_activation_quant()
    # Remove hooks after recalibration
    # Run a small batch to populate BN stats
    recalibrate_batchnorm(aq_model, train_loader, device, num_batches=20)


    # Final eval
    _, aq_val_acc = evaluate(aq_model, val_loader, nn.CrossEntropyLoss(), device)
    _, aq_test_acc = evaluate(aq_model, test_loader, nn.CrossEntropyLoss(), device)

    print(f"Final ACT-Q Val Acc  = {aq_val_acc:.2f}%")
    print(f"Final ACT-Q Test Acc = {aq_test_acc:.2f}%")

    # ----------------------------------------------
    # STEP 3 — MODEL SIZE COMPUTATION
    # ----------------------------------------------
    orig_size_fp32 = get_fp32_model_size_mb(orig_model)
    pruned_size_fp32 = get_pruned_model_size_mb(pruned_model)

    # Weight quantization size
    if weight_method == "fp8_e5m3":
        quantized_size = get_fp8_e5m3_size_mb(wq_model)
    elif weight_method == "fp8_e6m2":
        quantized_size = get_fp8_e6m2_size_mb(wq_model)
    elif weight_method in ["int8", "int6", "int4", "int2"]:
        bits = int(weight_method.replace("int", ""))
        quantized_size = get_weight_quant_size_mb(wq_model, bitwidth=bits)
    else:
        quantized_size = get_weight_quant_size_mb(wq_model, 8)  # fallback

    # ----------------------------------------------
    # RETURN SUMMARY
    # ----------------------------------------------
    return {
        "weight_method": weight_method,
        "activation_method": act_method,
        "val_acc": aq_val_acc,
        "test_acc": aq_test_acc,
        "orig_model_size": orig_size_fp32,
        "pruned_model_size": pruned_size_fp32,
        "compressed_model_size": quantized_size,
        "weight_compression_ratio": orig_size_fp32 / quantized_size,
        "pruned_compression_ratio": orig_size_fp32 / pruned_size_fp32
    }

