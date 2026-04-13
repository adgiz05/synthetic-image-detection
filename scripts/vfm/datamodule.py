"""
DataModule for VFM baseline training.

Loads images from CSV and preprocesses them using HuggingFace AutoImageProcessor
for center-crop + resize to native backbone resolution.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
from transformers import AutoImageProcessor
from typing import Optional, Dict, Any


class VFMDataset(Dataset):
    """
    Dataset for VFM baseline training.

    Loads images from CSV with columns:
    - image_path: relative path from data_root
    - label: 0 (real) or 1 (synthetic)
    - content_type, model, specific_model: metadata (optional)

    Args:
        csv_path: Path to CSV file
        data_root: Root directory for image paths
        processor: HuggingFace AutoImageProcessor for preprocessing
        return_metadata: Whether to return metadata (content_type, model, etc.)
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        processor: AutoImageProcessor,
        return_metadata: bool = False,
    ):
        self.data_root = Path(data_root)
        self.processor = processor
        self.return_metadata = return_metadata

        # Load CSV
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")

        # Verify columns
        required_cols = ["image_path", "label"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check label distribution
        label_counts = self.df["label"].value_counts()
        print(f"Label distribution: {dict(label_counts)}")

        # Derive a deterministic fallback size from processor config.
        # If size is unavailable, use a standard 224x224 image.
        self.fallback_size = self._get_fallback_size()

    def _get_fallback_size(self) -> tuple[int, int]:
        size = getattr(self.processor, "size", None)

        if isinstance(size, dict):
            if "height" in size and "width" in size:
                return int(size["width"]), int(size["height"])
            if "shortest_edge" in size:
                edge = int(size["shortest_edge"])
                return edge, edge
        elif isinstance(size, int):
            return size, size

        return 224, 224

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Load image
        img_path = self.data_root / row["image_path"]
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            # Keep training/eval running when corrupt or unreadable files appear.
            print(f"Warning: Failed to load image {img_path}: {e}. Using black fallback image.")
            image = Image.new("RGB", self.fallback_size, color=(0, 0, 0))

        # Preprocess using HuggingFace processor
        # IMPORTANT: AutoImageProcessor performs ONLY the following transformations:
        # 1. Resize to native resolution (e.g., 224x224 or 518x518)
        # 2. Center crop (deterministic)
        # 3. Normalization (model-specific mean/std)
        # NO random augmentations are applied (no random crop, flip, color jitter, etc.)
        # This matches the paper specification: "Images are resized and center-cropped
        # to the native resolution of each model without any additional data augmentation"
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)  # [3, H, W]

        # Prepare output
        output = {
            "image": pixel_values,
            "label": int(row["label"]),
        }

        # Add metadata if requested
        if self.return_metadata:
            metadata = {}
            for col in ["content_type", "model", "specific_model"]:
                if col in row:
                    metadata[col] = int(row[col])
            output["metadata"] = metadata

        return output


class VFMDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for VFM baseline training.

    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV (optional)
        test_csv: Path to test CSV (optional)
        data_root: Root directory for image paths
        backbone_name: HuggingFace model identifier (for processor)
        batch_size: Batch size per GPU
        num_workers: Number of dataloader workers
        return_metadata: Whether to return metadata
    """

    def __init__(
        self,
        train_csv: str,
        val_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
        data_root: str = "/home/adrian/synthetic-image-detection",
        backbone_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        batch_size: int = 128,
        num_workers: int = 4,
        return_metadata: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.data_root = data_root
        self.backbone_name = backbone_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_metadata = return_metadata

        # Load processor
        print(f"Loading image processor for: {backbone_name}")
        self.processor = AutoImageProcessor.from_pretrained(backbone_name)
        print(f"Native resolution: {self.processor.size}")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""

        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset = VFMDataset(
                csv_path=self.train_csv,
                data_root=self.data_root,
                processor=self.processor,
                return_metadata=self.return_metadata,
            )

            # Validation dataset
            if self.val_csv is not None:
                self.val_dataset = VFMDataset(
                    csv_path=self.val_csv,
                    data_root=self.data_root,
                    processor=self.processor,
                    return_metadata=self.return_metadata,
                )
            else:
                self.val_dataset = None
                print("Warning: No validation CSV provided")

        if stage == "test" or stage is None:
            if self.test_csv is not None:
                self.test_dataset = VFMDataset(
                    csv_path=self.test_csv,
                    data_root=self.data_root,
                    processor=self.processor,
                    return_metadata=self.return_metadata,
                )
            else:
                self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
