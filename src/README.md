# Synthetic Image Detection Library

A modular library for training and evaluating patch-based synthetic image detection models with attention aggregation.

## Features

- **Multi-Architecture Support**: Compatible with both Vision Transformers (ViT, DeiT, BEiT) and Convolutional architectures (ConvNeXt, ResNet)
- **Patch-based Processing**: Extracts and processes image patches with attention-based aggregation
- **Multi-Task Learning**: Binary classification with optional model and transform prediction
- **Augmentation**: Geometric degradation chain (resize + compression) with mask tracking
- **PyTorch Lightning**: Modern training infrastructure with W&B logging

## Installation

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Module Overview

### `models.py`

Core model architectures:

- **`FullImageModule`**: Main PyTorch Lightning module with multi-architecture backbone support
  - Supports ViT-style models (extracts CLS token)
  - Supports CNN-style models (spatial average pooling)
  - Automatic architecture detection
  - Attention-based patch aggregation
  - Multi-task heads (binary + optional model/transform prediction)

- **`AttnAggregator`**: Attention mechanism for aggregating patch embeddings

**Example Usage:**

```python
from src.models import FullImageModule

# ViT backbone
model_vit = FullImageModule(
    backbone_id="google/vit-base-patch16-224-in21k",
    num_classes=2,
    predict_model=True,
    num_model_classes=9,
    predict_transform=True,
    num_transform_classes=4,
    freeze_backbone=False,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=500,
    max_steps=10000
)

# ConvNeXt backbone
model_convnext = FullImageModule(
    backbone_id="facebook/convnextv2-large-22k-224",
    num_classes=2,
    predict_model=False,
    freeze_backbone=False,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=500,
    max_steps=10000
)
```

### `datasets.py`

Dataset implementations:

- **`FullImageDataset`**: Reads CSV with image paths, labels, model names, and benchmark info

**CSV Format:**

```csv
image_path,label,model,benchmark_id
data/images/img001.jpg,1,midjourney_v6,test_set
data/images/img002.jpg,0,real,validation
```

**Example Usage:**

```python
from src.datasets import FullImageDataset

dataset = FullImageDataset(
    csv_path="data/train.csv",
    model_to_int={"real": 0, "midjourney_v6": 1, ...},
    transform=None  # collator handles augmentation
)
```

### `collators.py`

Custom collators for batching:

- **`TrainFullImageDegradePatchCollator`**: Training collator with degradation augmentation
  - Probabilistic resize + JPEG compression chain
  - Tracks applied transforms via mask (0=none, 1=compression, 2=resize, 3=both)
  - Extracts sliding window patches
  - Zero-pads batch to max patch count

- **`ValFullImagePatchCollator`**: Validation collator (no augmentation)

**Example Usage:**

```python
from src.collators import TrainFullImageDegradePatchCollator

collator = TrainFullImageDegradePatchCollator(
    patch_size=224,
    stride=224,
    degrade_prob=0.5,
    resize_prob=0.5,
    compress_prob=0.5,
    max_resize_scale=0.5
)

# Returns dict with keys:
# - images: [B, N, 3, 224, 224] patches
# - attn_mask: [B, N] bool mask
# - labels: [B] binary labels
# - model_labels: [B] model indices
# - transform_mask: [B, N] transform type per patch
```

### `metrics.py`

Evaluation utilities:

- **`compute_binary_auc()`**: Computes AUC with fallback for edge cases
- **`safe_prf()`**: Precision/Recall/F1 with zero_division handling

### `patch_utils.py`

Image processing utilities:

- **`sliding_window_indices()`**: Generates patch coordinates with stride
- **`extract_patches_tensor_auto()`**: Extracts patches from PIL images with normalization

### `constants.py`

Shared constants:

- `IMAGENET_MEAN`, `IMAGENET_STD`: Normalization values
- `PROJECT`: Weights & Biases project name

## Usage in Training Scripts

### Training Script Example

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models import FullImageModule
from src.datasets import FullImageDataset
from src.collators import TrainFullImageDegradePatchCollator, ValFullImagePatchCollator

# Create module
module = FullImageModule(
    backbone_id="facebook/convnextv2-large-22k-224",  # or any HF model
    num_classes=2,
    predict_model=True,
    num_model_classes=9,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    max_steps=20000
)

# Create datasets
train_dataset = FullImageDataset("data/train.csv", model_to_int)
val_dataset = FullImageDataset("data/val.csv", model_to_int)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    collate_fn=TrainFullImageDegradePatchCollator(...),
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=8,
    collate_fn=ValFullImagePatchCollator(...),
    pin_memory=True
)

# Train
trainer = Trainer(max_steps=20000, ...)
trainer.fit(module, train_loader, val_loader)
```

## Architecture Support

### Vision Transformers (ViT-style)

Automatically detected by model ID keywords: `vit`, `deit`, `beit`

Feature extraction: CLS token (`outputs.last_hidden_state[:, 0, :]`)

**Supported Models:**
- `google/vit-base-patch16-224-in21k`
- `google/vit-large-patch16-224-in21k`
- `facebook/deit-base-distilled-patch16-224`
- `microsoft/beit-base-patch16-224`

### Convolutional Networks (CNN-style)

Automatically detected by model ID keywords: `convnext`, `resnet`, `efficientnet`

Feature extraction: Spatial average pooling over feature maps

**Supported Models:**
- `facebook/convnextv2-large-22k-224`
- `facebook/convnext-base-224`
- `microsoft/resnet-50`

### Adding Custom Architectures

The architecture detection happens automatically in `FullImageModule._is_vit_architecture()`:

1. **Keyword matching**: Checks model ID for known patterns
2. **Config fallback**: Checks if backbone has `config.num_attention_heads`

For custom architectures, ensure the model ID contains relevant keywords or the config properly indicates the architecture type.

## Multi-Task Learning

The model supports three prediction tasks:

1. **Binary Classification** (required): Real vs. Synthetic
2. **Model Prediction** (optional): Which generative model produced the image
3. **Transform Prediction** (optional): Which augmentation was applied (none/compression/resize/both)

Enable multi-task learning via constructor arguments:

```python
module = FullImageModule(
    backbone_id="facebook/convnextv2-large-22k-224",
    num_classes=2,
    predict_model=True,      # Enable model prediction
    num_model_classes=9,     # Number of generative models
    predict_transform=True,  # Enable transform prediction
    num_transform_classes=4  # 0=none, 1=compress, 2=resize, 3=both
)
```

Losses are automatically computed and weighted:
- `loss_main`: Binary cross-entropy
- `loss_model`: Model classification loss (if enabled)
- `loss_transform`: Transform classification loss (if enabled)

## Degradation Augmentation

The training collator applies a probabilistic augmentation chain:

```
1. Random Resize: Downscale to [0.5x, 1.0x] original size
2. Random JPEG Compression: Quality in [20, 95]
```

Each step is applied independently with configurable probabilities. A transform mask tracks which operations were applied to each patch (useful for transform prediction task).

**Configuration:**

```python
collator = TrainFullImageDegradePatchCollator(
    patch_size=224,
    stride=224,
    degrade_prob=0.5,         # Overall probability to apply any augmentation
    resize_prob=0.5,          # P(resize | degrade)
    compress_prob=0.5,        # P(compress | degrade)
    max_resize_scale=0.5      # Minimum scale for resize (0.5 = downsample to 50%)
)
```

## Patch Extraction

Images are divided into overlapping or non-overlapping patches using a sliding window:

- **Patch size**: Typically 224×224 (matches backbone input)
- **Stride**: Controls overlap (stride=224 for no overlap)
- **Attention mask**: Automatically generated to handle variable patch counts per image

The attention aggregator learns to weight important patches, producing a fixed-size image representation.

## Optimization

The module uses AdamW optimizer with:

- **Weight decay exclusion**: Bias and LayerNorm parameters excluded
- **Learning rate schedule**: Linear warmup + cosine decay
- **Gradient clipping**: Disabled by default (can enable via Trainer)

**Best Practices:**

- Start with LR ~1e-4 for pre-trained backbones
- Use warmup (5-10% of total steps)
- Adjust weight decay based on model size (0.01-0.05)

## Logging

Automatic logging to Weights & Biases:

- Training loss (main, model, transform)
- Validation metrics (accuracy, AUC, precision, recall, F1)
- Learning rate monitoring
- Per-benchmark evaluation (if benchmark_id provided in CSV)

Metrics are sanitized to remove invalid characters for W&B compatibility.

## Reproducibility

For deterministic training:

```python
import torch
import pytorch_lightning as pl

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note: Some operations (e.g., JPEG compression) may still introduce minor non-determinism.

## Contributing

When adding new features:

1. Keep modules focused (single responsibility)
2. Add type hints
3. Document complex algorithms
4. Test with both ViT and CNN backbones
5. Verify W&B logging works correctly

## License

See project root for license information.
