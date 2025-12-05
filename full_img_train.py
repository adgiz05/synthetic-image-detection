from src.datamodules import FullImageDataModule
from src.modules import FullImageModule, ImageModule
from src.configs import *

import os
import torch
import pytorch_lightning as pl
import argparse
from dataclasses import asdict
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
RESULTS_DIR = 'results/imaginet-full-img-cls'

def parse_args():
    parser = argparse.ArgumentParser(description='Whole-image fine-tuning')

    # --- dataset / patches ---
    parser.add_argument('--task', type=str, default='all', help='Classification task')
    parser.add_argument('--dataset_size', type=str, default='filtered', help='Dataset size (full, reduced, filtered)')
    parser.add_argument('--patch_size', type=int, default=224, help='Patch size')
    parser.add_argument('--max_patches', type=int, default=32, help='Max patches per image')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (smaller, because images -> seq of patches)')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')

    # --- model / optimizer / scheduler ---
    parser.add_argument('--model_id', type=str, default='google/vit-base-patch16-224-in21k', help='Backbone model')
    parser.add_argument('--checkpoint_path', type=str, default='imaginet-cls/datasetV2_filtered/checkpoints/epoch=99-step=94100.ckpt', help='Path to patch-level checkpoint (.ckpt)')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze ViT backbone')

    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer (sgd, adamw)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='linear+cosine', help='Scheduler type')
    parser.add_argument('--scheduler_skip', type=int, default=5, help='Epochs before switching to cosine')

    # --- losses ---
    parser.add_argument('--synthetic_weight', type=float, default=1.0, help='Synthetic loss weight')
    parser.add_argument('--model_weight', type=float, default=1.0, help='Model loss weight')
    parser.add_argument('--model_loss', type=str, default='generator', help="Model loss type ('generator' or 'specific_model')")

    # --- logging / training ---
    parser.add_argument('--project', type=str, default='imaginet-full-img-cls', help='W&B project name')
    parser.add_argument('--experiment', type=str, default='baseline', help='Experiment name')
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)

    args = parser.parse_args()

    # --- build config dataclasses ---
    return args


def setup_finetuning(args):
    # --- datamodule with dynamic stride ---
    datamodule = FullImageDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_config={"task": args.task},
        patch_size=args.patch_size,
        stride=None,              # auto stride
        max_patches=args.max_patches
    )

    # --- build the new aggregator model ---
    module = FullImageModule(
        backbone_path=args.checkpoint_path,
        optimizer_config={'name': args.optimizer, 'lr': args.lr, 'weight_decay': args.weight_decay},
        scheduler_config={'name': args.scheduler, 'max_epochs': args.max_epochs, 'scheduler_skip': args.scheduler_skip},
        loss_config={'synthetic_weight': args.synthetic_weight, 'model_weight': args.model_weight, 'model_loss': args.model_loss},
        freeze_backbone=args.freeze_backbone,
        aggregator="attn",
        aggregator_dim=512,
        device=args.device
    )

    # --- wandb logger ---
    logger = WandbLogger(name=args.experiment, project=args.project)
    profiler = AdvancedProfiler()

    # --- callbacks ---
    callbacks = [
        ModelCheckpoint(
            monitor='val/loss',
            mode='min',
            save_top_k=1,
        ),
        EarlyStopping(
            monitor='val/loss',
            mode='min',
            patience=args.early_stopping_patience,
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        accelerator='gpu',
        devices=[args.device],
        default_root_dir=f'{RESULTS_DIR}/{args.experiment}',
        profiler=profiler,
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    return datamodule, module, trainer


if __name__ == '__main__':
    args = parse_args()
    datamodule, module, trainer = setup_finetuning(args)
    trainer.fit(module, datamodule=datamodule)
