# VFM Baseline Training

Vision Foundation Model (VFM) baseline for synthetic image detection. Based on the paper specifications:
- **Frozen backbone**: DINOv3 weights are completely frozen
- **Linear head only**: Only the classification head is trained
- **AdamW optimizer**: lr=1e-3, weight_decay=0.01
- **Training setup**: effective batch_size=128 (batch_size=64 + gradient accumulation), 2 epochs
- **Preprocessing**: Images resized and center-cropped to native resolution (no augmentation)
- **QLoRA**: 4-bit quantization to fit 7B parameter model in 24GB VRAM

## Files

- `train.py`: Main training script with VFMClassifier Lightning module
- `datamodule.py`: VFMDataModule for loading data from CSV
- `run_train.sh`: Example training script

## Requirements

For QLoRA (4-bit quantization), the following packages are required:
- `bitsandbytes>=0.40.0` ✓ (already installed)
- `transformers>=4.30.0`
- `pytorch>=2.0.0`
- `pytorch-lightning`
- `wandb`

These are already available in the `synthetic-generation` conda environment.

## Usage

### Quick start (DINOv3)

```bash
# Default: uses GPU 5
bash scripts/vfm/run_train.sh

# Specify GPU (0-5)
bash scripts/vfm/run_train.sh 3  # Uses GPU 3

# Or set manually
GPU_ID=2 bash scripts/vfm/run_train.sh
```

The script uses `CUDA_DEVICE_ORDER=PCI_BUS_ID` to ensure consistent GPU ordering by PCI bus ID.

### Custom training

```bash
# With QLoRA (for 7B model) - RECOMMENDED
# Uses gradient accumulation to simulate batch_size=128
python scripts/vfm/train.py \
    --backbone facebook/dinov3-vit7b16-pretrain-lvd1689m \
    --train_csv data/dataset_v5/train.csv \
    --val_csv data/dataset_v5/val.csv \
    --data_root /home/adrian/synthetic-image-detection \
    --batch_size 64 \
    --accumulate_grad_batches 2 \
    --epochs 2 \
    --learning_rate 1e-3 \
    --output_dir runs/vfm/dinov3_vit7b16 \
    --precision 16-mixed \
    --device 5 \
    --wandb_project synthetic-detection-vfm \
    --wandb_name my_run \
    --use_qlora

# For smaller models without QLoRA
python scripts/vfm/train.py \
    --backbone facebook/dinov2-base \
    --batch_size 128 \
    --no_qlora
```

### W&B Logging

By default, training logs to Weights & Biases. Configure with:

```bash
--wandb_project my-project     # W&B project name (default: synthetic-detection-vfm)
--wandb_name my-run            # Run name (default: auto-generated from config)
--wandb_entity my-team         # W&B team/entity (optional)
--no_wandb                     # Disable W&B logging (use default logger)
```

Auto-generated run names follow the format: `{backbone}_bs{batch_size}_lr{learning_rate}`

### Available backbones

- `facebook/dinov3-vit7b16-pretrain-lvd1689m` (DINOv3, ~700M params) ← **default**
- `facebook/dinov2-small` (21M params, 224x224)
- `facebook/dinov2-base` (86M params, 224x224)
- `facebook/dinov2-large` (300M params, 224x224)
- `facebook/dinov2-giant` (1.1B params, 224x224)

## Data format

CSV with columns:
- `image_path`: relative path from data_root
- `label`: 0 (real) or 1 (synthetic)
- `content_type`, `model`, `specific_model`: optional metadata

Example:
```csv
image_path,label,content_type,model,specific_model
data/images/unsplash/riFTrh0K4pg.jpg,0,0,4,8
data/images/journeydb/3b6de56d.jpg,1,3,2,6
```

## Image preprocessing

Following the paper specification exactly:

> "Images are resized and center-cropped to the native resolution of each model without any additional data augmentation."

The implementation uses HuggingFace `AutoImageProcessor`, which applies:
1. **Resize** to the model's native resolution (e.g., 224×224 for DINOv2, 518×518 for some ViT variants)
2. **Center crop** (deterministic, no randomness)
3. **Normalization** with model-specific mean/std

**NO additional augmentations** are applied:
- ✅ No random crops
- ✅ No random flips
- ✅ No color jitter
- ✅ No random rotations
- ✅ No random erasing

This ensures fair comparison with the paper's VFM baseline.

## Memory considerations

**DINOv3-vit7b16 has 7B parameters** (not 700M). This is a very large model that requires special handling:

### With QLoRA (4-bit quantization) - RECOMMENDED
- Memory usage: ~12-14 GB with batch_size=64 (effective batch_size=128 with gradient accumulation)
- Training time: Slightly slower due to quantization overhead
- Quality: Minimal degradation vs full precision
- **This is the default configuration**

```bash
# QLoRA is enabled by default
# batch_size=64 + accumulate_grad_batches=2 → effective_batch_size=128 (same as paper)
bash scripts/vfm/run_train.sh

# Or explicitly:
python scripts/vfm/train.py --use_qlora --batch_size 64 --accumulate_grad_batches 2
```

### Gradient Accumulation

To match the paper's batch_size=128 while using less memory:
- Use `--batch_size 64 --accumulate_grad_batches 2` → effective batch_size = 128
- Use `--batch_size 32 --accumulate_grad_batches 4` → effective batch_size = 128

**How it works:**
- Gradients are accumulated over N batches before updating weights
- Memory usage is determined by `batch_size` (not effective batch_size)
- Training dynamics match the effective batch size
- Slightly slower (more forward passes per update) but uses less VRAM

### Without QLoRA (full precision) - NOT RECOMMENDED
- Memory usage: **>40 GB** (won't fit in 24GB GPUs)
- Only use with smaller backbones (dinov2-base, dinov2-small)

```bash
# Disable QLoRA (only for small models)
python scripts/vfm/train.py --no_qlora --backbone facebook/dinov2-base --batch_size 128
```

### QLoRA Parameters

- `--lora_r`: LoRA rank (default: 16). Higher = more parameters, better quality, more memory
- `--lora_alpha`: LoRA alpha (default: 32). Scaling factor for LoRA updates
- `--use_qlora`: Enable 4-bit quantization (default: True)
- `--no_qlora`: Disable QLoRA for small models
- `--accumulate_grad_batches`: Gradient accumulation steps (default: 1)

## Output

Checkpoints saved to `--output_dir`:
- `best-{epoch}-{val_acc}.ckpt`: Best model by validation accuracy
- `last.ckpt`: Last checkpoint

Logs available via TensorBoard:
```bash
tensorboard --logdir runs/vfm
```
