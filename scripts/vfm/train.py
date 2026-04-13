"""
VFM (Vision Foundation Model) baseline training.

Implementation based on paper specifications:
- Frozen backbone (DINOv3)
- Only linear head is trained
- AdamW optimizer, lr=1e-3
- Batch size 128, 2 epochs
- Images resized/center-cropped to native resolution
- No additional data augmentation
"""
import os
os.environ["HF_HOME"] = "/opt/huggingface/cache"  # Set HuggingFace cache directory
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Ensure consistent GPU ordering

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModel, AutoImageProcessor, BitsAndBytesConfig
from torchmetrics import Accuracy, AUROC
import argparse
from pathlib import Path
from datamodule import VFMDataModule


class VFMClassifier(pl.LightningModule):
    """
    VFM baseline: frozen backbone + trainable linear head.

    For large models (7B+ params), uses 4-bit quantization (QLoRA) to fit in 24GB VRAM.

    Args:
        backbone_name: HuggingFace model identifier (e.g., 'facebook/dinov3-vit7b16-pretrain-lvd1689m')
        learning_rate: Learning rate for AdamW optimizer
        num_classes: Number of output classes (default: 2 for binary)
        freeze_backbone: Whether to freeze the backbone (default: True)
        use_qlora: Use 4-bit quantization for the backbone (for large models)
        lora_r: LoRA rank (only if use_qlora=True)
        lora_alpha: LoRA alpha (only if use_qlora=True)
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        learning_rate: float = 1e-3,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        use_qlora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.use_qlora = use_qlora

        # Load backbone from HuggingFace
        print(f"Loading backbone: {backbone_name}")

        if use_qlora:
            # 4-bit quantization config for large models
            print("Using 4-bit quantization (QLoRA) for backbone")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            # Load model on single GPU (respects CUDA_VISIBLE_DEVICES)
            # device_map={"": 0} means "put everything on device 0" (which is the only visible device)
            self.backbone = AutoModel.from_pretrained(
                backbone_name,
                quantization_config=bnb_config,
                device_map={"": 0},  # Single GPU, not "auto" which uses all GPUs
            )

            # Note: With QLoRA, we don't add LoRA adapters to the backbone
            # We only train the linear head
            # The backbone is in 4-bit and frozen

        else:
            # Standard loading (for smaller models)
            self.backbone = AutoModel.from_pretrained(
                backbone_name,
                torch_dtype=torch.float32,
            )

        # Freeze backbone if requested
        if freeze_backbone:
            print("Freezing backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Get embedding dimension
        # For DINOv2/v3, the output is usually from the [CLS] token
        self.embed_dim = self.backbone.config.hidden_size
        print(f"Embedding dimension: {self.embed_dim}")

        # Linear classification head (always trained in FP32/BF16)
        self.head = nn.Linear(self.embed_dim, num_classes)

        # Metrics
        task = "binary" if num_classes == 2 else "multiclass"
        self.train_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_auroc = AUROC(task=task, num_classes=num_classes)

    def forward(self, pixel_values):
        """
        Forward pass through frozen backbone and linear head.

        Args:
            pixel_values: [B, 3, H, W] preprocessed images

        Returns:
            logits: [B, num_classes]
        """
        # Extract features from backbone
        with torch.set_grad_enabled(self.backbone.training):
            outputs = self.backbone(pixel_values)
            # DINOv2/v3 returns dict with 'last_hidden_state' [B, num_patches+1, D]
            # [CLS] token is at position 0
            features = outputs.last_hidden_state[:, 0]  # [B, D]

        # Keep dtype aligned with the classification head to avoid
        # Half/Float mismatches during matmul at inference time.
        features = features.to(dtype=self.head.weight.dtype)

        # Classification head
        logits = self.head(features)  # [B, num_classes]
        return logits

    def training_step(self, batch, batch_idx):
        """Training step."""
        images, labels = batch["image"], batch["label"]

        logits = self(images)

        # Binary cross-entropy loss
        if self.num_classes == 2:
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)
        else:
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)

        # Metrics
        acc = self.train_acc(preds, labels)

        # Logging
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, labels = batch["image"], batch["label"]

        logits = self(images)

        if self.num_classes == 2:
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)[:, 1]  # Probability of class 1
        else:
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)

        # Metrics
        self.val_acc(preds, labels)
        if self.num_classes == 2:
            self.val_auroc(probs, labels)
        else:
            self.val_auroc(probs, labels)

        # Logging
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/auroc", self.val_auroc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer as per paper."""
        # Only optimize the head parameters (backbone is frozen)
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        return optimizer

    def on_train_epoch_start(self):
        """Ensure backbone stays in eval mode."""
        if not self.hparams.freeze_backbone:
            return
        self.backbone.eval()


def main():
    parser = argparse.ArgumentParser(description="Train VFM baseline classifier")

    # Model args
    parser.add_argument(
        "--backbone",
        type=str,
        default="facebook/dinov3-vit7b16-pretrain-lvd1689m",
        help="HuggingFace backbone model identifier"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of output classes"
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (QLoRA) for large models (default: True for 7B model)"
    )
    parser.add_argument(
        "--no_qlora",
        action="store_true",
        help="Disable QLoRA (use full precision - requires more VRAM)"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (only used if use_qlora=True)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (only used if use_qlora=True)"
    )

    # Training args
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Accumulate gradients over N batches before updating weights (simulates larger batch size)"
    )

    # Data args
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to training CSV"
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=None,
        help="Path to validation CSV"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Path to test CSV"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/adrian/synthetic-image-detection",
        help="Root directory for image paths"
    )

    # System args
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/vfm",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Accelerator type"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=5,
        help="Number of devices"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Training precision"
    )

    # Logging args
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="synthetic-detection-vfm",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )

    args = parser.parse_args()

    # Configure GPU device BEFORE initializing model
    # This ensures the quantized model loads on the correct GPU
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        print(f"Setting CUDA_VISIBLE_DEVICES={args.device}")
        # After setting CUDA_VISIBLE_DEVICES, the only visible GPU is now "device 0"
        actual_device = 0
    else:
        actual_device = args.device

    # Handle QLoRA flag
    use_qlora = args.use_qlora and not args.no_qlora

    # Set seed
    pl.seed_everything(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = VFMClassifier(
        backbone_name=args.backbone,
        learning_rate=args.learning_rate,
        num_classes=args.num_classes,
        freeze_backbone=True,
        use_qlora=use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # Initialize datamodule
    datamodule = VFMDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        data_root=args.data_root,
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_metadata=False,
    )

    print("\n" + "="*50)
    print("VFM Baseline Training Configuration")
    print("="*50)
    print(f"Backbone: {args.backbone}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Accumulate grad batches: {args.accumulate_grad_batches}")
    print(f"Effective batch size: {args.batch_size * args.accumulate_grad_batches}")
    print(f"Epochs: {args.epochs}")
    print(f"Num classes: {args.num_classes}")
    print(f"Precision: {args.precision}")
    print(f"QLoRA (4-bit): {use_qlora}")
    if use_qlora:
        print(f"  LoRA rank: {args.lora_r}")
        print(f"  LoRA alpha: {args.lora_alpha}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Val CSV: {args.val_csv}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"W&B logging: {'Disabled' if args.no_wandb else 'Enabled'}")
    if not args.no_wandb:
        print(f"W&B project: {args.wandb_project}")
        print(f"W&B run name: {args.wandb_name or 'auto-generated'}")
    print("="*50 + "\n")

    # Initialize logger
    if args.no_wandb:
        logger = None
        print("W&B logging disabled. Using default logger.")
    else:
        # Auto-generate run name if not provided
        run_name = args.wandb_name
        if run_name is None:
            backbone_short = args.backbone.split('/')[-1]
            run_name = f"{backbone_short}_bs{args.batch_size}_lr{args.learning_rate}"

        logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            save_dir=args.output_dir,
            log_model=False,  # Don't upload checkpoints to W&B (can be large)
        )

        # Log hyperparameters
        logger.log_hyperparams({
            "backbone": args.backbone,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "effective_batch_size": args.batch_size * args.accumulate_grad_batches,
            "epochs": args.epochs,
            "num_classes": args.num_classes,
            "precision": args.precision,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "use_qlora": use_qlora,
            "lora_r": args.lora_r if use_qlora else None,
            "lora_alpha": args.lora_alpha if use_qlora else None,
        })

        print(f"W&B run initialized: {run_name}")

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="best-{epoch:02d}-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=[actual_device] if actual_device is not None else 1,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        default_root_dir=args.output_dir,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=0.25,
    )

    # Train
    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    print("\nTraining complete!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
