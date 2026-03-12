import os
import sys
import glob
import yaml
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# -------------------------------------------------------------------------
# Paths / env
# -------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['HF_HOME'] = '/opt/huggingface/cache'

# -------------------------------------------------------------------------
# Import library modules
# -------------------------------------------------------------------------

from src.constants import PROJECT
from src.models import FullImageModule
from src.datasets import FullImageDataset
from src.collators import TrainFullImageDegradePatchCollator, ValFullImagePatchCollator

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
                 max_patches: int = 32,
                 predict_model: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_path = train_path
        self.val_path = val_path
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.predict_model = predict_model

        # Train collator with degradations
        self.train_collate_fn = TrainFullImageDegradePatchCollator(
            patch_size=self.patch_size,
            max_patches=self.max_patches,
            normalize=True
        )
        # Val collator without degradations
        self.val_collate_fn = ValFullImagePatchCollator(
            patch_size=self.patch_size,
            max_patches=self.max_patches,
            normalize=True,
            return_benchmark=True,
        )

    def setup(self, stage=None):
        self.train_dataset = FullImageDataset(
            data_path=self.train_path,
            predict_model=self.predict_model,
            return_benchmark=False,
        )
        self.val_dataset = FullImageDataset(
            data_path=self.val_path,
            predict_model=self.predict_model,
            return_benchmark=True,
        )
        # Synchronize val model label mapping to use training set categories,
        # so both datasets share the same integer ↔ class assignment.
        if self.predict_model and hasattr(self.train_dataset, "model_label_names"):
            train_label_to_idx = {
                name: i for i, name in enumerate(self.train_dataset.model_label_names)
            }
            self.val_dataset.model_labels = [
                train_label_to_idx.get(m, 0) for m in self.val_dataset.model_label_raw
            ]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.val_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

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
    parser.add_argument("--predict_model", action="store_true", help="Also train a head to predict the generative model class")
    parser.add_argument("--predict_transform", action="store_true", help="Also train a head to predict the applied degradation transform")
    parser.add_argument("--lambda_model", type=float, default=1.0, help="Loss weight for model prediction head")
    parser.add_argument("--lambda_transform", type=float, default=1.0, help="Loss weight for transform prediction head")

    parser.add_argument("--max_epochs", type=int, default=25, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--monitor", type=str, default="val/loss", help="Metric to monitor for checkpointing")
    parser.add_argument("--mode", type=str, default="min", help="Mode for checkpointing (min or max)")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear LR warmup steps")
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

    base_output_dir = os.path.abspath(args.output_dir)

    # Detect checkpoint to resume from BEFORE creating a new output dir,
    # so we can re-use the existing run directory instead of a fresh one.
    ckpt_path = None
    if args.resume_if_possible:
        # Search for last.ckpt inside any existing run sub-directories
        candidates = sorted(
            glob.glob(os.path.join(base_output_dir, "*", "last.ckpt")),
            key=os.path.getmtime,
        )
        # Also check directly in base_output_dir
        direct_last = os.path.join(base_output_dir, "last.ckpt")
        if os.path.isfile(direct_last):
            candidates.append(direct_last)
        if candidates:
            ckpt_path = candidates[-1]
            output_dir = os.path.dirname(ckpt_path)
            run_name = os.path.basename(output_dir)
            print(f"[INFO] Resuming training from checkpoint: {ckpt_path}")
        else:
            print("[INFO] No checkpoint found; starting a new training run.")

    if ckpt_path is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_name)

    os.makedirs(output_dir, exist_ok=True)

    # Save args to args.yaml inside output_dir (use yaml.dump for valid YAML)
    args_yaml_path = os.path.join(output_dir, "args.yaml")
    with open(args_yaml_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    # Data
    datamodule = FullImageDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_path=args.train_path,
        val_path=args.validation_path,
        patch_size=args.patch_size,
        max_patches=args.max_patches,
        predict_model=args.predict_model,
    )

    # Setup to be able to inspect dataset metadata (e.g. num_model_classes)
    datamodule.setup(stage="fit")

    # Derive NUM_MODEL_CLASSES from the training set instead of hardcoding
    NUM_MODEL_CLASSES = (
        len(datamodule.train_dataset.model_label_names)
        if args.predict_model and hasattr(datamodule.train_dataset, "model_label_names")
        else 11
    )

    # Model
    model = FullImageModule(
        backbone_id=args.model_id,
        attn_dim=args.attn_dim,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        predict_model=args.predict_model,
        predict_transform=args.predict_transform,
        lambda_model=args.lambda_model,
        lambda_transform=args.lambda_transform,
        num_model_classes=NUM_MODEL_CLASSES,
    )

    # Logger (W&B) – logs go under output_dir
    logger = WandbLogger(
        project=PROJECT,
        name=run_name,
        save_dir=output_dir,
    )

    # Checkpoints directly under output_dir
    callbacks = [
        ModelCheckpoint(
            monitor=args.monitor,
            mode=args.mode,
            dirpath=output_dir,
            filename="best",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
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

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()