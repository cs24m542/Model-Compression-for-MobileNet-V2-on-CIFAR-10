# MobileNetV2 Model Compression on CIFAR-10  
### *Pruning • Weight Quantization • Activation Quantization • BN Recalibration*

This repository implements a complete **model compression pipeline** for MobileNetV2 fine-tuned on CIFAR-10.  
It explores structured pruning, multi-bit quantization (INT & FP8), and activation quantization while keeping accuracy as close as possible to the FP32 baseline.

The project demonstrates practical compression workflows aligned with modern deployment needs on edge devices and accelerators.

---

## 🚀 **Key Features**

### ✔ CIFAR-10 Training & Finetuning  
- Uses ImageNet-pretrained MobileNetV2  
- Head-only warmup + full fine-tuning  
- Reproducible training pipeline  

### ✔ Layer-Sensitive Pruning  
- Importance scores using L1 sensitivity  
- First & last layer protection  
- Smooth *cubic* sparsity schedule  
- Mask-based zero enforcement  
- Per-step fine-tuning  

### ✔ Weight Quantization  
Supports multiple quantization bit-widths:

| Method | Description |
|--------|-------------|
| `int2` | Symmetric 2-bit |
| `int4` | Symmetric 4-bit |
| `int6` | Symmetric 6-bit |
| `int8` | Standard INT8 |
| `fp8_e5m3` | FP8 (E5M3) simulated |
| `fp8_e6m2` | FP8 (E6M2) simulated |
| `kmeans` | Channelwise clustering |

### ✔ Activation Quantization
Available methods:
- INT8 with percentile clipping  
- FP8 (E5M3 / E6M2) per-channel scaling  
- GPU-safe percentile approximation  

### ✔ Compression Metrics  
The pipeline computes:
- FP32 model size  
- Pruned model size  
- Quantized model size  
- Total compression ratio  
- Accuracy drop at each stage  

### ✔ Optional W&B Integration  
- Parallel coordinate plots  
- Compression vs accuracy curves  
- Sparsity sweeps  
- Per-run configuration tracking  

---

## 📁 Project Structure



---

### 🎯 Ready-to-Insert README Block

Here is the final block you can drop into any README:

```md
## 📁 Project Structure


mobilenet-compression/
│
├── main.py                      # CLI orchestrator
├── data_transformation.py       # Dataloaders, transforms, dataset setup
├── training.py                  # Training, fine-tuning, accuracy functions
├── evaluation.py                # Evaluation utilities
├── compression.py               # Pruning, quantization, masks, size calculations
├── utils.py                     # Helper utilities
│
├── requirements.txt
└── README.md                    # ← You are here



---
```

# 🛠 **Installation**

### 1. Clone the repository
```bash
git clone https://github.com/<your_username>/mobilenet-compression.git
cd mobilenet-compression
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. (Optional) Enable W&B logging
wandb login

## Usage Guide

You can run the full workflow using command-line arguments in main.py.

## set seed
Default seed used is 42 for entire project, but you can configure something else with --seed option
```bash
python main.py --seed 123 --sparsity 0.5 --weight_method int8 --act_method fp8_e5m3 --wandb
```

## Prune the Model 

Example: prune to 60% sparsity, 8 pruning steps, 4 FT epochs each:
```bash 
python main.py \
    --prune \
    --target_sparsity 0.6 \
    --k_steps 8 \
    --finetune_epochs 4
```

## Outputs:

- Pruned model file in prune_checkpoints/

- CSV summary of pruning results

- Achieved sparsity + accuracy

## Run Weight Quantization Only
INT4:
```bash
python main.py --quantize-weights --weight_method int4

FP8 (E5M3):
python main.py --quantize-weights --weight_method fp8_e5m3
```
## Run Activation Quantization Only
INT8 activation quantization:
```bash
python main.py --quantize-acts --act_method int8
```
FP8:
```bash
python main.py --quantize-acts --act_method fp8_e6m2
```

## Full Compression Pipeline

(Pruning → Weight-Q → Activation-Q → Evaluate)
```bash
python main.py \
    --full-compression \
    --target_sparsity 0.5 \
    --weight_method int8 \
    --act_method fp8_e5m3
```

This returns:

- Final val/test accuracy

- Model sizes (FP32, pruned, quantized)

- Total compression ratio

## Output Format

Each run prints and logs:

### Accuracy Metrics
- Train Accuracy
- Validation Accuracy 
- Test Accuracy
- Accuracy Drop after Pruning
- Accuracy Drop after Quantization

### Model Sizes
- FP32 Model Size (MB)
- Pruned Model Size (MB)
- Quantized Model Size (MB)
- Total Compression Ratio (×)

### Artifacts Saved

- model_weights.pth (trained model)

- prune_checkpoints/*.pth (pruned versions)

- quantized_model.pth

- compression_summary.json (if enabled)

### W&B (optional)

- Parallel coordinate plot (accuracy vs sparsity vs size)

- Compression trends

- Layer sparsity histograms

## CLI Argument Summary
Argument	Description
-- train	Run training + finetuning
-- prune	Run pruning pipeline
--target_sparsity	Sparsity target (0–1)
--k_steps	Steps for gradual pruning
--finetune_epochs	FT epochs per pruning step
--quantize-weights	Perform weight quantization
--quantize-acts	Perform activation quantization
--weight_method	int2/int4/int6/int8/fp8_e5m3/fp8_e6m2
--act_method	int8/fp8_e5m3/fp8_e6m2
--full-compression	Run complete pipeline
## Notes

- Activation quantization uses histogram-based percentile clipping → prevents CUDA OOM

- First and last layers are protected during pruning for stability

- BN recalibration is essential after weight/activation quantization

- FP8 formats are simulated in PyTorch

## License

MIT License.

## Acknowledgments

This work was developed as part of the course CS6886W Assignment 3 — Model Compression.
It demonstrates practical pruning and quantization techniques suitable for deployment on modern ML accelerators.
