"""
Phase 1 contrastive training for the multi-scale tube forensics model.

Trains TubeContrastiveModule (phase=1) with:
  L = λ_auth · L_supcon_auth  +  λ_src · L_supcon_src  +  λ_decouple · L_decouple

Usage:
    python scripts/train_phase1.py \
        --train_path  data/train.csv \
        --val_path    data/val.csv \
        --output_dir  runs/phase1 \
        --device      0
"""

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

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"]           = "/opt/huggingface/cache"

# -------------------------------------------------------------------------
# Library imports
# -------------------------------------------------------------------------

from src.constants import PROJECT
from src.datasets import MultiScaleTubeDataset
from src.collators import MultiScaleTubeCollator, FastMultiScaleTubeCollator
from src.tube_model import TubeContrastiveModule

# -------------------------------------------------------------------------
# DataModule
# -------------------------------------------------------------------------

class TubeDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for multi-scale tube contrastive training.

    Supports two collator modes:
      - Legacy (MultiScaleTubeCollator): CPU preprocessing (residual + wavelet in collator)
      - Fast   (FastMultiScaleTubeCollator): GPU preprocessing (residual + wavelet in model)

    Train loader uses the selected collator (with all augmented views).
    Val   loader uses the same collator (augmented views are still needed
    to compute the Phase 1 contrastive val loss).
    """

    def __init__(
        self,
        train_path:    str,
        val_path:      str,
        root_dir:      str  = "",
        batch_size:    int  = 8,
        num_workers:   int  = 8,
        predict_model: bool = False,
        # Adaptive tube configuration
        max_tubes:    int       = 16,
        min_tubes:    int       = 4,
        overlap_ratio: float    = 0.25,
        scales:       list      = None,
        target_size:  int       = 128,
        num_views:    int       = 2,
        # Image size limits
        min_image_size: int = 256,
        max_image_size: int = 2048,
        # Collator mode
        use_fast_collator: bool = True,
    ):
        super().__init__()
        self.train_path     = train_path
        self.val_path       = val_path
        self.root_dir       = root_dir
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.predict_model  = predict_model

        self.max_tubes       = max_tubes
        self.min_tubes       = min_tubes
        self.overlap_ratio   = overlap_ratio
        self.scales          = scales or [64, 128, 256]
        self.target_size     = target_size
        self.num_views       = num_views
        self.min_image_size  = min_image_size
        self.max_image_size  = max_image_size
        self.use_fast_collator = use_fast_collator

        if use_fast_collator:
            # GPU preprocessing (residual/wavelet in model forward)
            self.collator = FastMultiScaleTubeCollator(
                max_tubes=self.max_tubes,
                min_tubes=self.min_tubes,
                overlap_ratio=self.overlap_ratio,
                scales=self.scales,
                target_size=self.target_size,
                num_views=self.num_views,
                min_image_size=self.min_image_size,
                max_image_size=self.max_image_size,
            )
            print("[INFO] Using FastMultiScaleTubeCollator (GPU preprocessing)")
        else:
            # CPU preprocessing (legacy, slower)
            self.collator = MultiScaleTubeCollator(
                max_tubes=self.max_tubes,
                min_tubes=self.min_tubes,
                overlap_ratio=self.overlap_ratio,
                scales=self.scales,
                target_size=self.target_size,
                num_views=self.num_views,
                normalize=True,
                min_image_size=self.min_image_size,
                max_image_size=self.max_image_size,
            )
            print("[INFO] Using MultiScaleTubeCollator (CPU preprocessing)")

    def setup(self, stage=None):
        self.train_dataset = MultiScaleTubeDataset(
            data_path=self.train_path,
            predict_model=self.predict_model,
            root_dir=self.root_dir,
        )
        self.val_dataset = MultiScaleTubeDataset(
            data_path=self.val_path,
            predict_model=self.predict_model,
            root_dir=self.root_dir,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1 contrastive training for the tube forensics model"
    )

    # Data
    parser.add_argument("--train_path",  type=str, required=True,  help="Path to training CSV")
    parser.add_argument("--val_path",    type=str, required=True,  help="Path to validation CSV")
    parser.add_argument("--output_dir",  type=str, required=True,  help="Base output directory")
    parser.add_argument("--root_dir",    type=str, default="",     help="Root dir prepended to relative image paths")
    parser.add_argument("--device",      type=int, required=True,  help="GPU device ID")

    # Adaptive tube configuration
    parser.add_argument("--max_tubes",      type=int,   default=16,   help="Maximum tubes per image (adaptive)")
    parser.add_argument("--min_tubes",      type=int,   default=4,    help="Minimum tubes per image")
    parser.add_argument("--overlap_ratio",  type=float, default=0.25, help="Target overlap ratio (0=no overlap, 0.5=half)")
    parser.add_argument("--scales",         type=int, nargs="+", default=[64, 128, 256], help="Crop sizes at each scale")
    parser.add_argument("--target_size",    type=int,          default=128,         help="Resize all crops to this size (px)")
    parser.add_argument("--num_views",      type=int,          default=2,           help="Views per tube (1 original + N-1 augmented)")
    parser.add_argument("--min_image_size", type=int,          default=256,         help="Skip images smaller than this")
    parser.add_argument("--max_image_size", type=int,          default=2048,        help="Resize images larger than this")
    parser.add_argument("--use_fast_collator", action="store_true", default=True, help="Use FastMultiScaleTubeCollator (GPU preprocessing)")
    parser.add_argument("--no_fast_collator",  action="store_true", help="Use legacy MultiScaleTubeCollator (CPU preprocessing)")

    # Model architecture
    parser.add_argument("--encoder_dim",        type=int,  default=256,  help="PatchEncoder output dim")
    parser.add_argument("--fused_dim",          type=int,  default=256,  help="FusionMLP output dim")
    parser.add_argument("--z_auth_dim",         type=int,  default=128,  help="Auth projection dim")
    parser.add_argument("--z_src_dim",          type=int,  default=128,  help="Src projection dim")
    parser.add_argument("--attn_dim",           type=int,  default=128,  help="MIL attention MLP dim")
    parser.add_argument("--pretrained_spatial", action="store_true",     help="ImageNet init for the 6-ch spatial encoder")
    parser.add_argument("--predict_model",      action="store_true",     help="Use model/generator labels")

    # Phase 1 loss weights
    parser.add_argument("--lambda_auth",     type=float, default=1.0,  help="Auth SupCon loss weight")
    parser.add_argument("--lambda_src_con",  type=float, default=0.5,  help="Src  SupCon loss weight")
    parser.add_argument("--lambda_decouple", type=float, default=0.01, help="Decoupling penalty weight")
    parser.add_argument("--temp_auth",       type=float, default=0.07, help="Temperature τ for auth SupCon")
    parser.add_argument("--temp_src",        type=float, default=0.07, help="Temperature τ for src  SupCon")

    # Training
    parser.add_argument("--lr",                      type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--warmup_steps",            type=int,   default=200,  help="Linear LR warm-up steps")
    parser.add_argument("--max_epochs",              type=int,   default=50,   help="Training epochs")
    parser.add_argument("--batch_size",              type=int,   default=8,    help="Batch size per GPU")
    parser.add_argument("--accumulate_grad_batches", type=int,   default=1,    help="Gradient accumulation steps")
    parser.add_argument("--num_workers",             type=int,   default=8,    help="DataLoader workers")
    parser.add_argument("--monitor",                 type=str,   default="val/loss", help="Metric to monitor for checkpointing")
    parser.add_argument("--mode",                    type=str,   default="min",      help="Checkpointing mode (min or max)")
    parser.add_argument("--resume_if_possible",      action="store_true", help="Resume from latest checkpoint in output_dir")

    return parser.parse_args()

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args = parse_args()
    pl.seed_everything(42)

    base_output_dir = os.path.abspath(args.output_dir)

    # ── Resume logic ──────────────────────────────────────────────────────
    ckpt_path = None
    if args.resume_if_possible:
        candidates = sorted(
            glob.glob(os.path.join(base_output_dir, "*", "last.ckpt")),
            key=os.path.getmtime,
        )
        direct = os.path.join(base_output_dir, "last.ckpt")
        if os.path.isfile(direct):
            candidates.append(direct)
        if candidates:
            ckpt_path  = candidates[-1]
            output_dir = os.path.dirname(ckpt_path)
            run_name   = os.path.basename(output_dir)
            print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
        else:
            print("[INFO] No checkpoint found; starting a new run.")

    if ckpt_path is None:
        run_name   = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, run_name)

    os.makedirs(output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    use_fast_collator = args.use_fast_collator and not args.no_fast_collator

    # Save hyper-parameters (include use_fast_collator for phase 2 to detect)
    saved_args = vars(args).copy()
    saved_args["use_fast_collator"] = use_fast_collator
    with open(os.path.join(output_dir, "args.yaml"), "w") as f:
        yaml.dump(saved_args, f, default_flow_style=False)

    datamodule = TubeDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        predict_model=args.predict_model,
        max_tubes=args.max_tubes,
        min_tubes=args.min_tubes,
        overlap_ratio=args.overlap_ratio,
        scales=args.scales,
        target_size=args.target_size,
        num_views=args.num_views,
        min_image_size=args.min_image_size,
        max_image_size=args.max_image_size,
        use_fast_collator=use_fast_collator,
    )
    datamodule.setup(stage="fit")

    # Derive num_src_classes from training set (needed for the source head used in phase 2)
    num_src_classes = None
    if args.predict_model and hasattr(datamodule.train_dataset, "model_label_names"):
        num_src_classes = len(datamodule.train_dataset.model_label_names)
        print(f"[INFO] num_src_classes = {num_src_classes}  ({datamodule.train_dataset.model_label_names})")

    # ── Model ─────────────────────────────────────────────────────────────
    model = TubeContrastiveModule(
        encoder_dim=args.encoder_dim,
        fused_dim=args.fused_dim,
        z_auth_dim=args.z_auth_dim,
        z_src_dim=args.z_src_dim,
        attn_dim=args.attn_dim,
        num_src_classes=num_src_classes,
        pretrained_spatial=args.pretrained_spatial,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        predict_model=args.predict_model,
        phase=1,
        target_size=args.target_size,
        lambda_auth=args.lambda_auth,
        lambda_src_con=args.lambda_src_con,
        lambda_decouple=args.lambda_decouple,
        temp_auth=args.temp_auth,
        temp_src=args.temp_src,
    )

    # Compile model for faster forward/backward (PyTorch 2.0+)
    try:
        model.model = torch.compile(model.model, mode="reduce-overhead")
        print("[INFO] Model compiled with torch.compile (reduce-overhead mode)")
    except Exception as e:
        print(f"[WARN] torch.compile failed: {e}. Skipping compilation.")

    # ── Logger ────────────────────────────────────────────────────────────
    logger = WandbLogger(
        project=f"{PROJECT}-phase1",
        name=run_name,
        save_dir=output_dir,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
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
    ]

    # ── Trainer ───────────────────────────────────────────────────────────
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
