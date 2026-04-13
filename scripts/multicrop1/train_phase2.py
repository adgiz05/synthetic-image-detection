"""
Phase 2 fine-tuning for binary classification (synthetic vs real).

Loads a phase 1 checkpoint and fine-tunes for the binary authentication task.
Supports selective layer freezing (encoders, fusion, projections).

Usage:
    python scripts/train_phase2.py \
        --phase1_ckpt runs/phase1/run_XXX/best.ckpt \
        --train_path  data/train.csv \
        --val_path    data/val.csv \
        --output_dir  runs/phase2 \
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
    Lightning DataModule for phase 2 binary classification.

    Uses num_views=1 (only original view, no augmentations) since phase 2
    is supervised classification without contrastive losses.

    Supports two collator modes:
      - Fast   (FastMultiScaleTubeCollator): GPU preprocessing (recommended)
      - Legacy (MultiScaleTubeCollator): CPU preprocessing
    """

    def __init__(
        self,
        train_path:    str,
        val_path:      str,
        root_dir:      str  = "",
        batch_size:    int  = 8,
        num_workers:   int  = 8,
        # Adaptive tube configuration
        max_tubes:    int       = 16,
        min_tubes:    int       = 4,
        overlap_ratio: float    = 0.25,
        scales:       list      = None,
        target_size:  int       = 128,
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

        self.max_tubes       = max_tubes
        self.min_tubes       = min_tubes
        self.overlap_ratio   = overlap_ratio
        self.scales          = scales or [64, 128, 256]
        self.target_size     = target_size
        self.min_image_size  = min_image_size
        self.max_image_size  = max_image_size
        self.use_fast_collator = use_fast_collator

        # Phase 2: only original view (no augmentations)
        if use_fast_collator:
            self.collator = FastMultiScaleTubeCollator(
                max_tubes=self.max_tubes,
                min_tubes=self.min_tubes,
                overlap_ratio=self.overlap_ratio,
                scales=self.scales,
                target_size=self.target_size,
                num_views=1,
                min_image_size=self.min_image_size,
                max_image_size=self.max_image_size,
            )
            print("[INFO] Using FastMultiScaleTubeCollator (GPU preprocessing)")
        else:
            self.collator = MultiScaleTubeCollator(
                max_tubes=self.max_tubes,
                min_tubes=self.min_tubes,
                overlap_ratio=self.overlap_ratio,
                scales=self.scales,
                target_size=self.target_size,
                num_views=1,
                normalize=True,
                min_image_size=self.min_image_size,
                max_image_size=self.max_image_size,
            )
            print("[INFO] Using MultiScaleTubeCollator (CPU preprocessing)")

    def setup(self, stage=None):
        self.train_dataset = MultiScaleTubeDataset(
            data_path=self.train_path,
            predict_model=False,  # Phase 2: only binary classification
            root_dir=self.root_dir,
        )
        self.val_dataset = MultiScaleTubeDataset(
            data_path=self.val_path,
            predict_model=False,
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
        description="Phase 2 binary classification fine-tuning"
    )

    # Phase 1 checkpoint (required)
    parser.add_argument("--phase1_ckpt", type=str, required=True,  help="Path to phase 1 checkpoint (.ckpt)")

    # Data (can be different from phase 1)
    parser.add_argument("--train_path",  type=str, required=True,  help="Path to training CSV")
    parser.add_argument("--val_path",    type=str, required=True,  help="Path to validation CSV")
    parser.add_argument("--output_dir",  type=str, required=True,  help="Base output directory")
    parser.add_argument("--root_dir",    type=str, default="",     help="Root dir prepended to relative image paths")
    parser.add_argument("--device",      type=int, required=True,  help="GPU device ID")

    # NOTE: Architecture and tube config are loaded from phase1 args.yaml automatically

    # Fine-tuning strategy
    parser.add_argument("--freeze_encoders",    action="store_true", default=True,  help="Freeze enc_spatial and enc_wavelet")
    parser.add_argument("--freeze_fusion",      action="store_true", default=False, help="Freeze FusionMLP")
    parser.add_argument("--freeze_projections", action="store_true", default=True,  help="Freeze projection heads (not used in phase 2)")
    parser.add_argument("--unfreeze_encoders",  action="store_true",                help="Override: unfreeze encoders (full fine-tuning)")
    parser.add_argument("--unfreeze_fusion",    action="store_true",                help="Override: unfreeze fusion (train it)")

    # Training
    parser.add_argument("--lr",                      type=float, default=1e-4, help="Peak learning rate (lower than phase 1)")
    parser.add_argument("--warmup_steps",            type=int,   default=100,  help="Linear LR warm-up steps")
    parser.add_argument("--max_epochs",              type=int,   default=20,   help="Training epochs")
    parser.add_argument("--batch_size",              type=int,   default=16,   help="Batch size per GPU")
    parser.add_argument("--accumulate_grad_batches", type=int,   default=1,    help="Gradient accumulation steps")
    parser.add_argument("--num_workers",             type=int,   default=8,    help="DataLoader workers")
    parser.add_argument("--monitor",                 type=str,   default="val/auc", help="Metric to monitor for checkpointing")
    parser.add_argument("--mode",                    type=str,   default="max",     help="Checkpointing mode (min or max)")
    parser.add_argument("--resume_if_possible",      action="store_true", help="Resume from latest checkpoint in output_dir")

    return parser.parse_args()

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args = parse_args()
    pl.seed_everything(42)

    # Validate phase1_ckpt exists
    if not os.path.isfile(args.phase1_ckpt):
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {args.phase1_ckpt}")

    # Load phase 1 hyperparameters from args.yaml
    phase1_dir = os.path.dirname(args.phase1_ckpt)
    phase1_args_path = os.path.join(phase1_dir, "args.yaml")
    if not os.path.isfile(phase1_args_path):
        raise FileNotFoundError(
            f"Phase 1 args.yaml not found: {phase1_args_path}\n"
            f"Make sure the checkpoint directory contains args.yaml from phase 1 training."
        )

    with open(phase1_args_path, "r") as f:
        phase1_args = yaml.safe_load(f)

    print(f"[INFO] Loading phase 1 hyperparameters from: {phase1_args_path}")

    # Extract architecture and data config from phase 1
    # Handle backward compatibility: old "num_tubes" → new "max_tubes"
    max_tubes      = phase1_args.get("max_tubes", phase1_args.get("num_tubes", 16))
    min_tubes      = phase1_args.get("min_tubes", 4)
    overlap_ratio  = phase1_args.get("overlap_ratio", 0.25)
    scales         = phase1_args["scales"]
    target_size    = phase1_args["target_size"]
    min_image_size = phase1_args.get("min_image_size", 256)
    max_image_size = phase1_args.get("max_image_size", 2048)
    encoder_dim    = phase1_args["encoder_dim"]
    fused_dim      = phase1_args["fused_dim"]
    z_auth_dim     = phase1_args["z_auth_dim"]
    z_src_dim      = phase1_args["z_src_dim"]
    attn_dim       = phase1_args["attn_dim"]
    pretrained_spatial = phase1_args.get("pretrained_spatial", False)

    print(f"[INFO] Architecture from phase 1:")
    print(f"  - Tubes: max={max_tubes}, min={min_tubes}, overlap={overlap_ratio}")
    print(f"  - Scales: {scales}, target_size={target_size}")
    print(f"  - Encoders: {encoder_dim}D, Fusion: {fused_dim}D")
    print(f"  - Projections: z_auth={z_auth_dim}D, z_src={z_src_dim}D")
    print(f"  - MIL attention: {attn_dim}D")

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

    # Save hyper-parameters (merge phase 2 args + phase 1 architecture)
    phase2_config = vars(args).copy()
    phase2_config.update({
        "max_tubes": max_tubes,
        "min_tubes": min_tubes,
        "overlap_ratio": overlap_ratio,
        "scales": scales,
        "target_size": target_size,
        "min_image_size": min_image_size,
        "max_image_size": max_image_size,
        "encoder_dim": encoder_dim,
        "fused_dim": fused_dim,
        "z_auth_dim": z_auth_dim,
        "z_src_dim": z_src_dim,
        "attn_dim": attn_dim,
        "pretrained_spatial": pretrained_spatial,
    })
    with open(os.path.join(output_dir, "args.yaml"), "w") as f:
        yaml.dump(phase2_config, f, default_flow_style=False)

    # ── Data ──────────────────────────────────────────────────────────────
    # Use fast collator by default (detect from phase 1 if available)
    use_fast_collator = phase1_args.get("use_fast_collator", True)
    datamodule = TubeDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_tubes=max_tubes,           # from phase 1
        min_tubes=min_tubes,           # from phase 1
        overlap_ratio=overlap_ratio,   # from phase 1
        scales=scales,                 # from phase 1
        target_size=target_size,       # from phase 1
        min_image_size=min_image_size, # from phase 1
        max_image_size=max_image_size, # from phase 1
        use_fast_collator=use_fast_collator,
    )
    datamodule.setup(stage="fit")

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"[INFO] Loading phase 1 checkpoint: {args.phase1_ckpt}")
    model = TubeContrastiveModule.load_from_checkpoint(
        args.phase1_ckpt,
        phase=2,               # Switch to phase 2
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        predict_model=False,   # Phase 2: only binary classification
        strict=False,          # Allow missing keys (e.g., phase1_loss state)
    )

    # Apply freezing strategy
    freeze_encoders = args.freeze_encoders and not args.unfreeze_encoders
    freeze_fusion   = args.freeze_fusion   and not args.unfreeze_fusion
    freeze_projections = args.freeze_projections  # Always True (not used in phase 2)

    model.freeze_layers(
        freeze_encoders=freeze_encoders,
        freeze_fusion=freeze_fusion,
        freeze_projections=freeze_projections,
    )

    print(f"[INFO] Fine-tuning strategy:")
    print(f"  - Encoders:    {'FROZEN' if freeze_encoders else 'TRAINABLE'}")
    print(f"  - Fusion MLP:  {'FROZEN' if freeze_fusion else 'TRAINABLE'}")
    print(f"  - Projections: FROZEN (not used in phase 2)")
    print(f"  - MIL + heads: TRAINABLE")

    # Compile model for faster forward/backward (PyTorch 2.0+)
    try:
        model.model = torch.compile(model.model, mode="reduce-overhead")
        print("[INFO] Model compiled with torch.compile (reduce-overhead mode)")
    except Exception as e:
        print(f"[WARN] torch.compile failed: {e}. Skipping compilation.")

    # ── Logger ────────────────────────────────────────────────────────────
    logger = WandbLogger(
        project=f"{PROJECT}-phase2",
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
