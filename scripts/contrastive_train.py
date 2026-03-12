"""
Contrastive training script for multi-scale tube-based forensic detection.

This script implements the architecture described in new_idea.md:
- Multi-scale tubes: patches at different scales centered at the same location
- Dual branch encoder: spatial/residual + wavelet/frequency
- Factorized embeddings: z_auth (authenticity) + z_src (source/generator)
- Contrastive learning with augmented views
- MIL aggregation for image-level predictions
"""

import os
from typing import Optional, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Local imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets import MultiScaleTubeDataset
from src.collators import MultiScaleTubeCollator


class MultiScaleTubeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for multi-scale tube contrastive learning.
    
    Encapsulates:
    - Train/val/test datasets using MultiScaleTubeDataset
    - Dataloaders with MultiScaleTubeCollator
    - All data configuration
    """
    
    def __init__(
        self,
        # Data paths
        train_csv: str,
        val_csv: str,
        test_csv: Optional[str] = None,
        root_dir: str = "",
        # Task configuration
        predict_model: bool = False,
        # Tube configuration (for collator)
        num_tubes: int = 8,
        scales: List[int] = [64, 128, 256],
        target_size: int = 128,
        num_views: int = 2,
        # Degradation params for training views
        jpeg_prob: float = 0.5,
        jpeg_quality_range: Tuple[int, int] = (70, 95),
        resize_prob: float = 0.3,
        resize_range: Tuple[float, float] = (0.8, 1.2),
        blur_prob: float = 0.2,
        blur_sigma_range: Tuple[float, float] = (0.5, 2.0),
        sharpen_prob: float = 0.2,
        sharpen_strength_range: Tuple[float, float] = (0.5, 2.0),
        noise_prob: float = 0.3,
        noise_std_range: Tuple[float, float] = (0.01, 0.05),
        # Image size constraints
        min_image_size: int = 256,
        max_image_size: int = 2048,
        # DataLoader params
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        # Other
        normalize: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Data paths
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.root_dir = root_dir
        
        # Task
        self.predict_model = predict_model
        
        # Tube & collator config
        self.num_tubes = num_tubes
        self.scales = scales
        self.target_size = target_size
        self.num_views = num_views
        
        # Degradation params
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.resize_prob = resize_prob
        self.resize_range = resize_range
        self.blur_prob = blur_prob
        self.blur_sigma_range = blur_sigma_range
        self.sharpen_prob = sharpen_prob
        self.sharpen_strength_range = sharpen_strength_range
        self.noise_prob = noise_prob
        self.noise_std_range = noise_std_range
        
        # Image size constraints
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        
        # DataLoader config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.normalize = normalize
        
        # Will be populated in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing.
        
        Args:
            stage: 'fit', 'validate', 'test', or None for all
        """
        if stage == "fit" or stage is None:
            self.train_dataset = MultiScaleTubeDataset(
                data_path=self.train_csv,
                predict_model=self.predict_model,
                root_dir=self.root_dir,
            )
            
            self.val_dataset = MultiScaleTubeDataset(
                data_path=self.val_csv,
                predict_model=self.predict_model,
                root_dir=self.root_dir,
            )
            
            print(f"[DataModule] Train dataset: {len(self.train_dataset)} samples")
            print(f"[DataModule] Val dataset: {len(self.val_dataset)} samples")
            
            if self.predict_model and hasattr(self.train_dataset, 'model_label_names'):
                print(f"[DataModule] Model classes: {len(self.train_dataset.model_label_names)}")
                print(f"[DataModule] Model names: {self.train_dataset.model_label_names}")
        
        if stage == "test" or stage is None:
            if self.test_csv is not None:
                self.test_dataset = MultiScaleTubeDataset(
                    data_path=self.test_csv,
                    predict_model=self.predict_model,
                    root_dir=self.root_dir,
                )
                print(f"[DataModule] Test dataset: {len(self.test_dataset)} samples")
    
    def _create_collator(self, is_train: bool = True):
        """
        Create collator for train or val/test.
        
        For training: full augmentations
        For val/test: could use reduced augmentations or no augmentations
        """
        if is_train:
            return MultiScaleTubeCollator(
                num_tubes=self.num_tubes,
                scales=self.scales,
                target_size=self.target_size,
                num_views=self.num_views,
                jpeg_prob=self.jpeg_prob,
                jpeg_quality_range=self.jpeg_quality_range,
                resize_prob=self.resize_prob,
                resize_range=self.resize_range,
                blur_prob=self.blur_prob,
                blur_sigma_range=self.blur_sigma_range,
                sharpen_prob=self.sharpen_prob,
                sharpen_strength_range=self.sharpen_strength_range,
                noise_prob=self.noise_prob,
                noise_std_range=self.noise_std_range,
                normalize=self.normalize,
                min_image_size=self.min_image_size,
                max_image_size=self.max_image_size,
            )
        else:
            # For validation/test: use reduced augmentation
            # (or no augmentation depending on your strategy)
            return MultiScaleTubeCollator(
                num_tubes=self.num_tubes,
                scales=self.scales,
                target_size=self.target_size,
                num_views=1,  # Single view for validation
                jpeg_prob=0.0,  # No degradations
                resize_prob=0.0,
                blur_prob=0.0,
                sharpen_prob=0.0,
                noise_prob=0.0,
                normalize=self.normalize,
                min_image_size=self.min_image_size,
                max_image_size=self.max_image_size,
            )
    
    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._create_collator(is_train=True),
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._create_collator(is_train=False),
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self):
        """Create test dataloader."""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._create_collator(is_train=False),
            persistent_workers=self.num_workers > 0,
        )


def main():
    """
    Main training function.
    
    TODO: Implement model, training loop, etc.
    For now, this is a skeleton to test the DataModule.
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train CSV")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--test_csv", type=str, default=None, help="Path to test CSV")
    parser.add_argument("--root_dir", type=str, default="", help="Root directory for image paths")
    parser.add_argument("--predict_model", action="store_true", help="Add model prediction head")
    
    # Tube configuration
    parser.add_argument("--num_tubes", type=int, default=8, help="Number of tubes per image")
    parser.add_argument("--scales", type=int, nargs="+", default=[64, 128, 256], help="Scales for multi-scale tubes")
    parser.add_argument("--target_size", type=int, default=128, help="Target size for patches")
    parser.add_argument("--num_views", type=int, default=2, help="Number of augmented views per patch")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")
    
    # Model configuration (TODO: will be extended when model is implemented)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create DataModule
    datamodule = MultiScaleTubeDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        root_dir=args.root_dir,
        predict_model=args.predict_model,
        num_tubes=args.num_tubes,
        scales=args.scales,
        target_size=args.target_size,
        num_views=args.num_views,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Setup (for testing)
    datamodule.setup("fit")
    
    # Test loading a batch
    print("\n[Testing DataModule] Loading a training batch...")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Tubes shape: {batch['tubes'].shape}")  # [B, N_tubes, K_scales, V_views, C, P, P]
    print(f"Tube centers shape: {batch['tube_centers'].shape}")  # [B, N_tubes, 2]
    print(f"Labels shape: {batch['labels'].shape}")  # [B]
    if 'model_labels' in batch:
        print(f"Model labels shape: {batch['model_labels'].shape}")
    
    print("\n[Testing DataModule] Loading a validation batch...")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    
    print(f"Val batch keys: {val_batch.keys()}")
    print(f"Val tubes shape: {val_batch['tubes'].shape}")
    print(f"Val tube centers shape: {val_batch['tube_centers'].shape}")
    print(f"Val labels shape: {val_batch['labels'].shape}")
    
    print("\n[SUCCESS] DataModule is working correctly!")
    print("\nNext steps:")
    print("  1. Implement dual-branch encoder (spatial/residual + wavelet)")
    print("  2. Implement embedding factorization (z_auth + z_src)")
    print("  3. Implement contrastive losses")
    print("  4. Implement MIL aggregation for image-level prediction")


if __name__ == "__main__":
    main()
