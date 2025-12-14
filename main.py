#!/usr/bin/env python3
import argparse
import torch
import wandb
import pandas as pd
from pathlib import Path

# === Import your modularized files ===
from data_transformation import get_dataloaders
from training import (
	create_model,
	train_model
)
from evaluation import evaluate
from compression import (
    Do_pruning,
    full_compression_pipeline,
    get_fp32_model_size_mb,
    get_pruned_model_size_mb
)
from utility import (
	compression_ratio,
	set_seed
)
	
g_config={
        "epochs": 20,
        "batch_size": 256,
        "learning_rate_FT": 0.001 ,
        "learning_rate_HT": 0.01,
        "momentum":0.9,
        "weight_decay":5e-4,
        "optimizer": "SGD",
	    "head_epochs": 5,
        "model": "MobileNetV2"
    }


# -----------------------------------------------------------
#  CLI ARGUMENTS
# -----------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="MobileNetV2 Pruning + Quantization Pipeline")
    parser.add_argument("--train", action="store_true",
                        help="Enable training mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--prune", type=float, default=None,
                        help="Target sparsity (e.g., 0.5). If not set, pruning is skipped.")

    parser.add_argument("--quant_w", type=str, default="int8",
                        choices=["int8", "int6", "int4", "int2", "fp8_e5m3", "fp8_e6m2"],
                        help="Weight quantization method")

    parser.add_argument("--quant_a", type=str, default="int8",
                        choices=["int8", "fp8_e5m3", "fp8_e6m2"],
                        help="Activation quantization method")

    parser.add_argument("--epochs_ft", type=int, default=4,
                        help="Fine-tuning epochs per pruning step")

    parser.add_argument("--steps", type=int, default=8,
                        help="Pruning steps (K steps)")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--load_ckpt", type=str, default="./model_weights.pth",
                        help="Path to MobileNet finetuned checkpoint")

    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")

    return parser.parse_args()


# -----------------------------------------------------------
#  MAIN PIPELINE
# -----------------------------------------------------------
def main():
    args = get_args()
    set_seed(args.seed)

    # -------------------------
    #  Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    #  Load CIFAR-10
    # -------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, args.batch_size,seed=args.seed
    )

    # -------------------------
    #  Load Model (ImageNet-pretrained architecture)
    # -------------------------

    model = create_model(num_classes=10, pretrained=False).to(device)
    # -------------------------
    #  W&B Init (optional)
    # -------------------------
    if args.wandb:
        wandb.init(
            project="mobilenetv2-cifar10-compression",
            config={
                "prune": args.prune,
                "quant_w": args.quant_w,
                "quant_a": args.quant_a,
                "epochs_ft": args.epochs_ft,
                "steps": args.steps,
            }
        )

    if(args.train):
        del model
        model = create_model(num_classes=10, pretrained=True).to(device)
    # -------------------------
        best_state_dict, history, best_val_acc = train_model(
	        model,
            train_loader,
            val_loader, 
            g_config, 
            device,
            args.wandb
        )
        #this will overwrite if we have saved weights already
        torch.save(best_state_dict, "model_weights.pth")
    # Load fine-tuned checkpoint
    print(f"Loading checkpoint: {args.load_ckpt}")
    ckpt = torch.load(args.load_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    results_table = []
    # -------------------------
    #  Evaluate baseline
    # -------------------------
    baseline_loss, baseline_acc = evaluate(model, val_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"\nBaseline Val Acc = {baseline_acc:.2f}%")

    orig_model_size = get_fp32_model_size_mb(model)
    print(f"Baseline FP32 Model Size = {orig_model_size:.3f} MB")


    # -------------------------
    #  Step 1: PRUNING
    # -------------------------
    Sparcity = 0
    if args.prune is not None:
        Sparcity = args.prune
        print(f"\n=== Running pruning: target sparsity = {args.prune} ===")

        df_prune, pruned_model = Do_pruning(
            model_loaded=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            target_sparsity=args.prune,
            K_steps=args.steps,
            finetune_epochs_per_step=args.epochs_ft,
            baseline_val_acc=baseline_acc,
	    seed=args.seed
        )

        pruned_size = get_pruned_model_size_mb(pruned_model)
        print(f"Pruned Model Size = {pruned_size:.3f} MB")

    else:
        print("\nSkipping pruning — using original model.")
        pruned_model = model

    # -------------------------
    #  Step 2: QUANTIZATION
    # -------------------------
    print(f"\n=== Quantization: W = {args.quant_w}, A = {args.quant_a} ===")

    quant_results = full_compression_pipeline(
        orig_model=model,
        pruned_model=pruned_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        weight_method=args.quant_w,
        act_method=args.quant_a
    )
    
    results_table.append({
                "Sparcity":Sparcity,
                "Weight Method": args.quant_w,
                "Activation Method": args.quant_a,
                "Val Acc": quant_results["val_acc"],
                "Test Acc": quant_results["test_acc"],
                "Pruned Compression Ratio": compression_ratio(orig_model_size,pruned_size),
                "Total Compression Ratio": compression_ratio(orig_model_size,quant_results["compressed_model_size"]),
                "Orig_Model_Size": orig_model_size,
                "Pruned_Model_Size": pruned_size,
                "Compressed_Model_Size":quant_results["compressed_model_size"]
            })
    if args.wandb:
        wandb.log({
                "Sparcity":Sparcity,
                "weight_method": args.quant_w,
                "activation_method": args.quant_a,
                "val_accuracy": quant_results["val_acc"],
                "model_size_mb": quant_results["compressed_model_size"],
                "compression_ratio": compression_ratio(orig_model_size, quant_results["compressed_model_size"])
            })
    
    print("\n=== FINAL RESULTS ===")
    df_acc = pd.DataFrame(results_table)
    print(df_acc)

    # -------------------------
    #  Log results to W&B
    # -------------------------

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()

