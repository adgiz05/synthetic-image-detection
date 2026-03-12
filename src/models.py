"""Model architectures for full-image classification."""

import math
from typing import Dict, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel

from .metrics import compute_binary_auc
import numpy as np


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
    """
    Full-image classification module with patch-based processing.
    
    Supports:
    - Main binary classification (synthetic vs real)
    - Optional auxiliary heads for model prediction and transform prediction
    """
    
    def __init__(
        self,
        backbone_id: str = "google/vit-base-patch16-224-in21k",
        attn_dim: int = 128,
        lr: float = 5e-4,
        warmup_steps: int = 100,
        predict_model: bool = False,
        predict_transform: bool = False,
        lambda_model: float = 1.0,
        lambda_transform: float = 1.0,
        num_model_classes: int = None,
        num_transform_classes: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.predict_model = predict_model
        self.predict_transform = predict_transform
        self.lambda_model = lambda_model
        self.lambda_transform = lambda_transform

        self.lr = lr
        self.backbone_id = backbone_id

        # Backbone: Load pre-trained model
        self.backbone = AutoModel.from_pretrained(backbone_id)
        
        # Get hidden dimension (handle both hidden_size and hidden_sizes)
        if hasattr(self.backbone.config, 'hidden_size'):
            hidden_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, 'hidden_sizes'):
            # For models like ConvNeXt, use the last hidden size
            hidden_dim = self.backbone.config.hidden_sizes[-1]
        else:
            raise ValueError(f"Cannot determine hidden dimension for backbone {backbone_id}")

        # Detect backbone architecture type
        # ViT-like: has attention layers and produces sequence outputs with CLS token
        # CNN-like (ConvNeXt, ResNet, etc.): produces spatial feature maps
        self.is_vit_like = self._is_vit_architecture()
        
        # For CNN backbones, we need adaptive pooling to convert spatial features to vectors
        if not self.is_vit_like:
            self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Attention-based aggregator
        self.aggregator = AttnAggregator(hidden_dim, attn_dim)

        # Main binary classification head (synthetic vs non-synthetic)
        self.classifier = nn.Linear(hidden_dim, 2)

        # Optional model head
        if self.predict_model:
            if num_model_classes is None:
                raise ValueError("num_model_classes must be provided when predict_model=True")
            self.model_classifier = nn.Linear(hidden_dim, num_model_classes)
        else:
            self.model_classifier = None

        # Optional transform head (0: none, 1: compression, 2: resize, 3: both)
        if self.predict_transform:
            self.transform_classifier = nn.Linear(hidden_dim, num_transform_classes)
        else:
            self.transform_classifier = None

        # Losses
        self.loss_main = nn.CrossEntropyLoss()
        self.loss_model = nn.CrossEntropyLoss() if self.predict_model else None
        self.loss_transform = nn.CrossEntropyLoss() if self.predict_transform else None

        # Buffer to store validation step outputs for epoch-level AUC
        self.val_outputs = []

    def _is_vit_architecture(self) -> bool:
        """
        Detect if the backbone is a ViT-like architecture (has CLS token)
        or a CNN-like architecture (spatial feature maps).
        
        Returns:
            True if ViT-like (uses CLS token), False if CNN-like (needs pooling)
        """
        backbone_name = self.backbone_id.lower()
        
        # ViT-like models (have CLS token)
        vit_keywords = ['vit', 'deit', 'beit', 'swin', 'dino']
        if any(kw in backbone_name for kw in vit_keywords):
            return True
        
        # CNN-like models (spatial feature maps, need pooling)
        cnn_keywords = ['convnext', 'resnet', 'efficientnet', 'mobilenet', 'regnet']
        if any(kw in backbone_name for kw in cnn_keywords):
            return False
        
        # Fallback: check if model has 'num_attention_heads' in config (ViT characteristic)
        return hasattr(self.backbone.config, 'num_attention_heads')

    def _extract_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch embeddings from backbone, handling both ViT and CNN architectures.
        
        Args:
            x: Input tensor [B*N, C, H, W] where B=batch, N=num_patches
            
        Returns:
            patch_emb: [B*N, D] patch embeddings
        """
        # Forward through backbone
        # Only pass interpolate_pos_encoding for ViT-like models
        if self.is_vit_like:
            outputs = self.backbone(pixel_values=x, return_dict=True, interpolate_pos_encoding=True)
        else:
            outputs = self.backbone(pixel_values=x, return_dict=True)
        
        if self.is_vit_like:
            # ViT-like: Extract CLS token from sequence
            # last_hidden_state shape: [B*N, seq_len, D]
            # CLS token is at position 0
            patch_emb = outputs.last_hidden_state[:, 0, :]  # [B*N, D]
        else:
            # CNN-like: Global average pooling over spatial dimensions
            # last_hidden_state shape: [B*N, D, H', W'] for ConvNeXt or similar
            features = outputs.last_hidden_state
            
            # Handle different output formats
            if len(features.shape) == 4:  # [B*N, D, H', W']
                # Apply spatial pooling
                patch_emb = self.spatial_pool(features).squeeze(-1).squeeze(-1)  # [B*N, D]
            elif len(features.shape) == 3:  # [B*N, seq_len, D]
                # Some models might output sequence format, take mean over sequence
                patch_emb = features.mean(dim=1)  # [B*N, D]
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        return patch_emb

    def forward(self, images: torch.Tensor, attn_mask: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            images:   [B, N, C, H, W] (patches from collator)
            attn_mask:[B, N] (bool) True where a patch exists

        Returns:
            dict with:
                - pooled:          [B, D] pooled features
                - attn:            [B, N] patch attention weights
                - logits_main:     [B, 2]
                - logits_model:    [B, M] or None
                - logits_transform:[B, T] or None
        """
        B, N, C, H, W = images.shape

        # Flatten patches to feed into backbone
        x = images.view(B * N, C, H, W)  # [B*N, C, H, W]

        # Extract patch embeddings (handles both ViT and CNN architectures)
        patch_emb = self._extract_patch_embeddings(x)  # [B*N, D]
        D = patch_emb.size(-1)

        # Reshape back to [B, N, D]
        patch_emb = patch_emb.view(B, N, D)

        # Aggregate over patches with attention
        pooled, attn = self.aggregator(patch_emb, attn_mask)  # [B, D], [B, N]

        # Main head
        logits_main = self.classifier(pooled)

        # Optional heads
        logits_model = self.model_classifier(pooled) if self.predict_model else None
        logits_transform = self.transform_classifier(pooled) if self.predict_transform else None

        return {
            "pooled": pooled,
            "attn": attn,
            "logits_main": logits_main,
            "logits_model": logits_model,
            "logits_transform": logits_transform,
        }

    def training_step(self, batch, batch_idx):
        """
        Multi-task training:
          - Main task: synthetic vs non-synthetic (mandatory)
          - Optional tasks:
                * model prediction (model_label)
                * transform prediction (transforms)
        """
        images = batch["images"]
        mask = batch["attn_mask"]
        labels = batch["labels"]

        out = self(images, mask)
        logits_main = out["logits_main"]

        # Main loss
        loss_main = self.loss_main(logits_main, labels)
        loss = loss_main

        # Log main loss as "train/loss" and "train/loss_main"
        self.log("train/loss", loss_main, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_main", loss_main, prog_bar=False, on_step=True, on_epoch=True)

        # Optional model loss (only for synthetic images)
        if self.predict_model and "model_label" in batch and out["logits_model"] is not None:
            model_labels = batch["model_label"]
            # Only compute loss for synthetic images (labels == 1)
            synthetic_mask = (labels == 1)
            if synthetic_mask.sum() > 0:
                loss_model = self.loss_model(out["logits_model"][synthetic_mask], model_labels[synthetic_mask])
                loss = loss + self.lambda_model * loss_model
                self.log("train/loss_model", loss_model, prog_bar=False, on_step=True, on_epoch=True)
            else:
                # No synthetic images in batch, log 0
                self.log("train/loss_model", 0.0, prog_bar=False, on_step=True, on_epoch=True)

        # Optional transform loss (train collator provides 'transforms')
        if self.predict_transform and "transforms" in batch and out["logits_transform"] is not None:
            transform_labels = batch["transforms"]
            loss_transform = self.loss_transform(out["logits_transform"], transform_labels)
            loss = loss + self.lambda_transform * loss_transform
            self.log("train/loss_transform", loss_transform, prog_bar=False, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        mask = batch["attn_mask"]
        labels = batch["labels"]
        benchmarks = batch.get("benchmarks", None)

        out = self(images, mask)
        logits_main = out["logits_main"]

        # Main val loss
        loss_main = self.loss_main(logits_main, labels)

        preds = logits_main.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        # Log standard metrics for the main task
        self.log("val/loss", loss_main, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/acc", acc, prog_bar=True, on_epoch=True, on_step=False)

        # Optional: log aux losses in validation if labels are available
        if self.predict_model and "model_label" in batch and out["logits_model"] is not None:
            model_labels = batch["model_label"]
            # Only compute loss for synthetic images (labels == 1)
            synthetic_mask = (labels == 1)
            if synthetic_mask.sum() > 0:
                loss_model = self.loss_model(out["logits_model"][synthetic_mask], model_labels[synthetic_mask])
                self.log("val/loss_model", loss_model, prog_bar=False, on_epoch=True, on_step=False)
            else:
                self.log("val/loss_model", 0.0, prog_bar=False, on_epoch=True, on_step=False)

        if self.predict_transform and "transforms" in batch and out["logits_transform"] is not None:
            loss_transform = self.loss_transform(out["logits_transform"], batch["transforms"])
            self.log("val/loss_transform", loss_transform, prog_bar=False, on_epoch=True, on_step=False)

        # For AUC we only care about the main head
        probs = torch.softmax(logits_main, dim=-1)[:, 1]  # probability of class 1

        out_dict = {
            "loss": loss_main.detach(),
            "labels": labels.detach(),
            "probs": probs.detach(),
            "benchmarks": benchmarks,
        }

        # Store for epoch-end aggregation
        self.val_outputs.append(out_dict)

        return out_dict

    def on_validation_epoch_end(self):
        """
        Aggregate validation outputs to compute global and per-benchmark AUC
        for the main binary task (synthetic vs non-synthetic).
        """
        outputs = self.val_outputs
        self.val_outputs = []  # reset for next epoch

        if self.classifier.out_features != 2:
            return

        all_labels = []
        all_scores = []
        all_benchmarks = []

        for out in outputs:
            labels = out["labels"]
            probs = out["probs"]
            benchmarks = out["benchmarks"]

            if probs is None:
                continue

            all_labels.append(labels.cpu().numpy())
            all_scores.append(probs.cpu().numpy())

            if benchmarks is None:
                all_benchmarks.extend(["unknown"] * labels.size(0))
            else:
                all_benchmarks.extend(list(benchmarks))

        if not all_labels:
            return

        labels_arr = np.concatenate(all_labels, axis=0)
        scores_arr = np.concatenate(all_scores, axis=0)

        # Global AUC
        global_auc = compute_binary_auc(labels_arr, scores_arr)
        if not np.isnan(global_auc):
            self.log("val/auc", global_auc, prog_bar=True, on_epoch=True)

        # Per-benchmark AUC
        import re
        unique_benchmarks = sorted(set(all_benchmarks))
        for bm in unique_benchmarks:
            if bm is None:
                continue
            bm_mask = np.array([b == bm for b in all_benchmarks])
            bm_labels = labels_arr[bm_mask]
            bm_scores = scores_arr[bm_mask]

            if bm_labels.size == 0:
                continue

            bm_auc = compute_binary_auc(bm_labels, bm_scores)
            if np.isnan(bm_auc):
                continue

            metric_name = re.sub(r"[^a-zA-Z0-9_/]", "_", f"bm/auc_{bm}")
            self.log(metric_name, bm_auc, prog_bar=False, on_epoch=True)

    def configure_optimizers(self):
        # Exclude bias and normalization parameters from weight decay (standard ViT practice)
        no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias",
                    "layer_norm.weight", "layer_norm.bias"}
        param_groups = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-2,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        # Linear warmup + cosine decay
        warmup_steps = self.hparams.warmup_steps
        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Any]:
        """Prediction step for testing/inference."""
        images = batch["images"]
        mask = batch["attn_mask"]

        out = self(images, mask)
        logits = out["logits_main"]

        probs = torch.softmax(logits, dim=-1)  # [B,2]
        preds = probs.argmax(dim=-1)           # [B]
        conf = probs.max(dim=-1).values        # [B]

        return {
            "image_paths": batch.get("image_paths", []),
            "abs_paths": batch.get("abs_paths", []),
            "benchmarks": batch.get("benchmarks", []),
            "labels": batch.get("labels", torch.tensor([-1] * images.size(0))).detach().cpu(),
            "preds": preds.detach().cpu(),
            "conf": conf.detach().cpu(),
            "prob0": probs[:, 0].detach().cpu(),
            "prob1": probs[:, 1].detach().cpu(),
            "is_fallback": batch.get("is_fallback", [False] * images.size(0)),
            "load_errors": batch.get("load_errors", [""] * images.size(0)),
        }
