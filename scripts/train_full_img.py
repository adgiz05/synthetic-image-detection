import os
import sys
import numpy as np
import pandas as pd

import argparse
from dataclasses import dataclass, field
import glob
from datetime import datetime
from typing import List, Tuple

import random
import io
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from transformers import AutoModel, AutoConfig
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime

# -------------------------------------------------------------------------
# Paths / env
# -------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['HF_HOME'] = '/opt/huggingface/cache'

PROJECT = 'lawwwing-full-img-cls'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class FullImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, predict_model: bool = False):
        data = pd.read_csv(data_path)

        self.image_paths = data["image_path"].tolist()
        # If you need both label and model, adapt this part
        self.labels = data["label"].tolist() if not predict_model else data[["label", "model"]].values

    def __len__(self) -> int:
        return len(self.image_paths)

    def _blank_image_fallback(self, h: int = 512, w: int = 512) -> Image.Image:
        """Return a blank RGB image if something goes wrong when loading."""
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = self._blank_image_fallback()

        label = self.labels[idx]
        
        return {"image": img, "label": label, "path": path}


# -------------------------------------------------------------------------
# Patch extraction helpers
# -------------------------------------------------------------------------

def sliding_window_indices(H: int, W: int, patch: int, stride: int) -> List[Tuple[int, int]]:
    """
    Generate top-left indices for a sliding window over an HxW image.
    We ensure at least one index (0, 0) if the image is smaller.
    """
    ys = list(range(0, max(H - patch, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - patch, 0) + 1, stride)) or [0]
    return [(y, x) for y in ys for x in xs]


def extract_patches_tensor_auto(x: torch.Tensor,
                                patch: int,
                                max_patches: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract patches with an automatically chosen stride so that
    we get roughly <= max_patches. If we still get more, randomly
    subsample.

    Args:
        x:          Tensor [C,H,W]
        patch:      Patch size
        max_patches:Max number of patches (approximate)

    Returns:
        patches: [N,C,patch,patch]
        coords:  [N,2] with normalized centers (cy, cx) in [0,1]
    """
    C, H, W = x.shape

    # Choose stride based on a target grid size ~ sqrt(max_patches)
    grid = max(1, int(math.sqrt(max_patches)))
    stride = max(1, min(H, W) // grid)

    coords, patches = [], []

    # Sliding window over the full image
    for (y, x0) in sliding_window_indices(H, W, patch, stride):
        y2, x2 = y + patch, x0 + patch
        yy2, xx2 = min(y2, H), min(x2, W)

        crop = x[:, y:yy2, x0:xx2]

        # Pad if near border (right / bottom)
        pad_h = patch - crop.shape[1]
        pad_w = patch - crop.shape[2]
        if pad_h > 0 or pad_w > 0:
            # pad format: (left, right, top, bottom)
            crop = F.pad(crop, (0, pad_w, 0, pad_h))

        patches.append(crop)

        # Center (in pixels)
        cy_pix = min(y + patch / 2, H)
        cx_pix = min(x0 + patch / 2, W)

        # Normalize by original H,W
        cy = cy_pix / max(H, 1)
        cx = cx_pix / max(W, 1)
        coords.append([cy, cx])

    # Fallback if something weird happens (should not occur with current logic)
    if not patches:
        patches = [x]
        coords = [[0.5, 0.5]]

    patches = torch.stack(patches, dim=0)
    coords = torch.tensor(coords, dtype=torch.float32)

    # If we got too many patches, randomly subsample
    N = patches.shape[0]
    if N > max_patches:
        idx = torch.randperm(N)[:max_patches]
        patches = patches[idx]
        coords = coords[idx]

    return patches, coords


# -------------------------------------------------------------------------
# Collator: degradation + tiling
# -------------------------------------------------------------------------

class FullImageDegradePatchCollator:
    """
    Collator that:
      1) Applies a geometric chain of degradations (resize + compression)
      2) Extracts multiple patches from the degraded image using an automatic stride
      3) Returns:
           - batched patches padded to max length
           - per-image label
           - per-image transform mask (0..3)
           - per-patch coords and attention mask
           - optionally original (pre-degradation) tensors

    Transform mask (per original image):
        0 = no transform
        1 = compression applied at least once
        2 = resize applied at least once
        3 = both compression and resize were applied
    """

    def __init__(
        self,
        # degradation chain
        step_continue_prob: float = 0.5,
        max_steps: int = 10,
        compression_prob: float = 0.5,
        resize_prob: float = 0.5,
        jpeg_ratio: float = 0.5,
        min_quality: int = 70,
        max_quality: int = 90,
        min_resize_ratio: float = 0.8,
        max_resize_ratio: float = 1.0,
        # patch extraction
        patch_size: int = 224,
        max_patches: int = 32,
        # options
        return_original: bool = False,
        normalize: bool = True,
    ):
        # Degradation params
        self.step_continue_prob = step_continue_prob
        self.max_steps = max_steps
        self.compression_prob = compression_prob
        self.resize_prob = resize_prob
        self.jpeg_ratio = jpeg_ratio
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.min_resize_ratio = min_resize_ratio
        self.max_resize_ratio = max_resize_ratio

        # Patch params
        self.patch_size = patch_size
        self.max_patches = max_patches

        self.return_original = return_original

        # Transforms for tensor conversion / normalization
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) if normalize else None

    # ------------- Degradation functions -------------

    def degrade_once(self, img: Image.Image) -> Tuple[Image.Image, int]:
        """
        Apply a single degradation step.

        Returns:
            new_img, transform_mask

        Where transform_mask bits are:
            1 = compression happened
            2 = resize happened
        """
        mask = 0

        # Resize
        if random.random() < self.resize_prob:
            w, h = img.size
            ratio = random.uniform(self.min_resize_ratio, self.max_resize_ratio)
            new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))
            img = TF.resize(img, (new_h, new_w))
            mask |= 2  # resize bit

        # Compression
        if random.random() < self.compression_prob:
            fmt = "JPEG" if random.random() < self.jpeg_ratio else "WEBP"
            quality = random.randint(self.min_quality, self.max_quality)
            buf = io.BytesIO()
            img.save(buf, format=fmt, quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            mask |= 1  # compression bit

        return img, mask

    def degrade_image_chain(self, img: Image.Image) -> Tuple[Image.Image, int]:
        """
        Apply a geometric chain of degradation steps:
            while random() < step_continue_prob AND steps < max_steps:
                apply degrade_once()

        Returns:
            degraded_img, total_mask (0..3)
        """
        total_mask = 0
        steps = 0

        while True:
            img, m = self.degrade_once(img)
            total_mask |= m
            steps += 1

            if steps >= self.max_steps or random.random() > self.step_continue_prob:
                break

        return img, total_mask

    def __call__(self, batch):
        """
        Args:
            batch: list of dicts {"image": PIL.Image or Tensor, "label": int, "path": str}

        Returns:
            output dict with:
                - images:     [B, Nmax, C, P, P]
                - coords:     [B, Nmax, 2]
                - attn_mask:  [B, Nmax] (True where a patch exists)
                - labels:     [B]
                - transforms: [B] (0..3)
                - paths:      list of length B with image paths
                - originals:  [B, C, H, W] (optional)
        """
        seqs = []          # list of [Ni, C, P, P]
        coords_list = []   # list of [Ni, 2]
        lengths = []       # list of Ni
        labels = []        # list of labels
        transforms = []    # per-image transform code
        originals = []     # optional pre-degradation tensors
        paths = []         # list of image paths

        for item in batch:
            img = item["image"]
            label = item["label"]
            path = item.get("path", None)

            # Ensure we start from a PIL image for degradation
            if isinstance(img, torch.Tensor):
                img = TF.to_pil_image(img)

            # Optionally store the original (pre-degradation) tensor
            if self.return_original:
                orig_t = self.to_tensor(img)
                if self.normalize is not None:
                    orig_t = self.normalize(orig_t)
                originals.append(orig_t)

            # Apply degradation chain
            degraded_img, t_mask = self.degrade_image_chain(img)
            transforms.append(t_mask)
            paths.append(path)

            # Convert degraded image to tensor
            t = self.to_tensor(degraded_img)
            if self.normalize is not None:
                t = self.normalize(t)

            # Always use automatic patch extraction
            patches, c = extract_patches_tensor_auto(
                t,
                self.patch_size,
                self.max_patches
            )

            seqs.append(patches)
            coords_list.append(c)
            lengths.append(patches.shape[0])
            labels.append(label)

        # Pad to batch with max sequence length
        B = len(batch)
        Nmax = max(lengths)
        C, P = seqs[0].shape[1], seqs[0].shape[2]

        images = torch.zeros(B, Nmax, C, P, P, dtype=seqs[0].dtype)
        coord_t = torch.zeros(B, Nmax, 2, dtype=coords_list[0].dtype)
        attn_mask = torch.zeros(B, Nmax, dtype=torch.bool)

        for i, (p, c, l) in enumerate(zip(seqs, coords_list, lengths)):
            images[i, :l] = p
            coord_t[i, :l] = c
            attn_mask[i, :l] = True

        # Convert labels / transforms to tensors (assuming integer labels)
        labels = torch.as_tensor(labels, dtype=torch.long)
        transforms = torch.as_tensor(transforms, dtype=torch.long)

        output = {
            "images": images,        # [B, Nmax, C, P, P]
            "coords": coord_t,       # [B, Nmax, 2]
            "attn_mask": attn_mask,  # [B, Nmax]
            "labels": labels,        # [B]
            "transforms": transforms # [B]
        }

        if self.return_original and len(originals) > 0:
            output["originals"] = torch.stack(originals, dim=0)  # [B, C, H, W]

        return output


# -------------------------------------------------------------------------
# DataModule
# -------------------------------------------------------------------------

class FullImageDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_path: str,
                 val_path: str,
                 batch_size: int = 8,
                 num_workers: int = 8,
                 patch_size: int = 224,
                 max_patches: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_path = train_path
        self.val_path = val_path
        self.patch_size = patch_size
        self.max_patches = max_patches

        self.collate_fn = FullImageDegradePatchCollator(
            patch_size=self.patch_size,
            max_patches=self.max_patches,
            normalize=True
        )

    def setup(self, stage=None):
        self.train_dataset = FullImageDataset(data_path=self.train_path)
        self.val_dataset = FullImageDataset(data_path=self.val_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn
        )


# -------------------------------------------------------------------------
# Attention aggregator + LightningModule
# -------------------------------------------------------------------------

class AttnAggregator(nn.Module):
    """Simple attention pooling over a variable-length sequence of patch embeddings."""

    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        # Small MLP to produce attention logits
        self.fc1 = nn.Linear(hidden_dim, attn_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(attn_dim, 1)

    def forward(self, patch_embeddings: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            patch_embeddings: [B, N, D]
            mask:             [B, N] (bool) True = valid patch

        Returns:
            pooled: [B, D] aggregated representation
            attn:   [B, N] attention weights per patch
        """
        # Compute raw attention scores
        # [B, N, D] -> [B, N, attn_dim] -> [B, N, 1]
        scores = self.fc2(self.tanh(self.fc1(patch_embeddings))).squeeze(-1)  # [B, N]

        # Mask out padded positions
        scores = scores.masked_fill(~mask, -1e9)

        # Normalize to attention weights
        attn = torch.softmax(scores, dim=-1)  # [B, N]

        # Weighted sum of patch embeddings
        pooled = torch.sum(patch_embeddings * attn.unsqueeze(-1), dim=1)  # [B, D]

        return pooled, attn


class FullImageModule(pl.LightningModule):
    def __init__(
        self,
        backbone_id: str = "google/vit-base-patch16-224-in21k",
        attn_dim: int = 128,
        lr: float = 5e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Backbone: ViT without classification head
        self.backbone = AutoModel.from_pretrained(backbone_id)
        hidden_dim = self.backbone.config.hidden_size

        # Only attention-based aggregator
        self.aggregator = AttnAggregator(hidden_dim, attn_dim)

        # Classification head
        self.classifier = nn.Linear(hidden_dim, 2)

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, images: torch.Tensor, attn_mask: torch.Tensor):
        """
        Args:
            images:   [B, N, C, H, W] (patches from collator)
            attn_mask:[B, N] (bool) True where a patch exists

        Returns:
            logits: [B, num_labels]
            attn:   [B, N] attention over patches
        """
        B, N, C, H, W = images.shape

        # Flatten patches to feed into ViT
        x = images.view(B * N, C, H, W)  # [B*N, C, H, W]

        # HuggingFace ViT expects `pixel_values`
        outputs = self.backbone(pixel_values=x, return_dict=True, interpolate_pos_encoding=True)

        # Take CLS token for each patch
        patch_emb = outputs.last_hidden_state[:, 0, :]  # [B*N, D]
        D = patch_emb.size(-1)

        # Reshape back to [B, N, D]
        patch_emb = patch_emb.view(B, N, D)

        # Aggregate over patches with attention
        pooled, attn = self.aggregator(patch_emb, attn_mask)  # [B, D], [B, N]

        # Final classification
        logits = self.classifier(pooled)  # [B, num_labels]

        return logits, attn

    def training_step(self, batch, batch_idx):
        """
        Expects batch from FullImageDegradePatchCollator:
          batch["images"]: [B, N, C, P, P]
          batch["attn_mask"]: [B, N]
          batch["labels"]: [B]
        """
        images = batch["images"]
        mask = batch["attn_mask"]
        labels = batch["labels"]

        logits, attn = self(images, mask)
        loss = self.loss_fn(logits, labels)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        mask = batch["attn_mask"]
        labels = batch["labels"]

        logits, attn = self(images, mask)
        loss = self.loss_fn(logits, labels)

        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

# -------------------------------------------------------------------------
# CLI arguments
# -------------------------------------------------------------------------

def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, required=True, help="Path to training csv")
    parser.add_argument("--validation_path", type=str, required=True, help="Path to validation csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=int, required=True, help="GPU device ID")

    parser.add_argument("--model_id", type=str, default="google/vit-base-patch16-224-in21k", help="Backbone model ID")
    parser.add_argument("--patch_size", type=int, default=224, help="Patch size")
    parser.add_argument("--max_patches", type=int, default=32, help="Max patches per image")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--attn_dim", type=int, default=128, help="Attention dimension")

    parser.add_argument("--max_epochs", type=int, default=25, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--resume_if_possible", action="store_true", help="If set, resume training from latest checkpoint found in output_dir")

    args = parser.parse_args()
    return args

# -------------------------------------------------------------------------
# Main: training loop
# -------------------------------------------------------------------------
def main():
    args = parse_args()

    # Set seeds for reproducibility
    pl.seed_everything(42)

    # Output dir: everything will be stored directly here
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save args to args.yaml inside output_dir
    args_yaml_path = os.path.join(output_dir, "args.yaml")
    with open(args_yaml_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # Data
    datamodule = FullImageDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_path=args.train_path,
        val_path=args.validation_path,
        patch_size=args.patch_size,
        max_patches=args.max_patches,
    )

    # Model
    model = FullImageModule(
        backbone_id=args.model_id,
        attn_dim=args.attn_dim,
        lr=args.lr,
    )

    # Run name for logging
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    # Logger (W&B) – logs go under output_dir
    logger = WandbLogger(
        project=PROJECT,
        name=run_name,
        save_dir=output_dir,
    )

    # Checkpoints directly under output_dir
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            dirpath=output_dir,
            filename="best-{epoch:02d}",
            save_top_k=1,
            save_last=True,
        ),
        # EarlyStopping(
        #     monitor="val/loss",
        #     mode="min",
        #     patience=10
        # )
    ]
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        precision="bf16",
        devices=[args.device],
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=output_dir,
    )

    # Detect checkpoint to resume from (if requested)
    ckpt_path = None
    if args.resume_if_possible:
        # Prefer 'last.ckpt' if it exists (exact last training state)
        last_ckpt = os.path.join(output_dir, "last.ckpt")
        if os.path.isfile(last_ckpt):
            ckpt_path = last_ckpt
        else:
            # Otherwise, pick the most recently modified .ckpt file
            ckpts = sorted(
                glob.glob(os.path.join(output_dir, "*.ckpt")),
                key=os.path.getmtime,
            )
            if ckpts:
                ckpt_path = ckpts[-1]

        if ckpt_path is not None:
            print(f"[INFO] Resuming training from checkpoint: {ckpt_path}")
        else:
            print("[INFO] No checkpoint found in output_dir; starting a new training run.")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()