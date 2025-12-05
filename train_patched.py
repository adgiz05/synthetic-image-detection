from src.datamodules import ImageDataModule
from src.modules import ImageModule
from src.configs import *

import os

import pytorch_lightning as pl
import argparse
from dataclasses import asdict
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

RESULTS_DIR = 'results/classification'

def parse_args():
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--task', type=str, default='all', help='Loss task')
    parser.add_argument('--size', type=int, default=224, help='Image size')
    parser.add_argument('--augmentation', type=str, default='patched', help='Augmentation (patched, none)')

    parser.add_argument('--dataset_size', type=str, default='full', help='Dataset size (full, reduced)')

    parser.add_argument('--no_transformations', action='store_false', dest='transformations', help='Whether to apply transformations')    
    parser.add_argument('--downscaling_prob', type=float, default=0.3, help='Downscaling probability')
    parser.add_argument('--resize_prob', type=float, default=0.3, help='Resize probability')
    parser.add_argument('--compression_prob', type=float, default=0.3, help='Compression probability')
    parser.add_argument('--blur_noise_prob', type=float, default=0.1, help='Blur and noise probability')
    parser.add_argument('--color_prob', type=float, default=0.1, help='Color probability')
    parser.add_argument('--texture_prob', type=float, default=0.1, help='Texture probability')
    parser.add_argument('--local_artifacts_prob', type=float, default=0.1, help='Local artifacts probability')
    parser.add_argument('--rotation_prob', type=float, default=0.1, help='Rotation probability')
    parser.add_argument('--flip_prob', type=float, default=0.1, help='Flip probability')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')

    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer (sgd, adam, adamw)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')

    parser.add_argument('--scheduler', type=str, default='linear+cosine', help='Scheduler (linear, cosine, linear+cosine)')
    parser.add_argument('--scheduler_skip', type=int, default=10, help='Number of epochs to skip from linear to cosine')

    parser.add_argument('--synthetic_weight', type=float, default=1.0, help='Weight for synthetic loss')
    parser.add_argument('--model_weight', type=float, default=1.0, help='Weight for model loss')
    parser.add_argument('--model_loss', type=str, default='generator', help="Model loss type ('generator' or 'specific_model')")

    parser.add_argument('--model_id', type=str, default='google/vit-base-patch16-224-in21k', help='Backbone model')

    parser.add_argument('--project', type=str, default='imaginet-cls', help='Project name')
    parser.add_argument('--experiment', type=str, default='baseline', help='Experiment name')

    parser.add_argument('--early_stopping_patience', type=int, default=50, help='Early stopping patience')

    parser.add_argument('--max_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--precision', type=str, default='bf16-mixed', help='Precision (32, 16, bf16-mixed)')
    parser.add_argument('--device', type=int, default=0, help='GPU to train in')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5, help='Check val every n epochs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulate grad batches')

    args = parser.parse_args()

    augmentation_config = AugmentationConfig(
        transformations=args.transformations,
        downscaling_prob=args.downscaling_prob,
        resize_prob=args.resize_prob,
        compression_prob=args.compression_prob,
        blur_noise_prob=args.blur_noise_prob,
        color_prob=args.color_prob,
        texture_prob=args.texture_prob,
        local_artifacts_prob=args.local_artifacts_prob,
        rotation_prob=args.rotation_prob,
        flip_prob=args.flip_prob,
    )
    dataset_config = ImageDatasetConfig(task=args.task, size=args.size, augmentation=args.augmentation, augmentation_config=augmentation_config, dataset_size=args.dataset_size)
    datamodule_config = ImageDataModuleConfig(batch_size=args.batch_size, num_workers=args.num_workers, train_dataset_config=dataset_config, val_dataset_config=dataset_config)
    optimizer_config = ImageOptimizerConfig(name=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
    scheduler_config = ImageSchedulerConfig(name=args.scheduler, max_epochs=args.max_epochs, scheduler_skip=args.scheduler_skip)
    loss_config = ImageLossConfig(synthetic_weight=args.synthetic_weight, model_weight=args.model_weight, model_loss=args.model_loss)  
    module_config = ImageModuleConfig(model_id=args.model_id, optimizer_config=optimizer_config, scheduler_config=scheduler_config, loss_config=loss_config)
    logger_config = ImageLoggerConfig(project=args.project, experiment_name=args.experiment)
    callbacks_config = ImageCallbacksConfig(early_stopping_patience=args.early_stopping_patience)
    
    return ImageConfig(
        datamodule_config=datamodule_config,
        module_config=module_config,
        logger_config=logger_config,
        callbacks_config=callbacks_config,
        max_epochs=args.max_epochs,
        precision=args.precision,
        device=args.device,
        default_root_dir=RESULTS_DIR,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

def setup_pretraining(config):
    datamodule = ImageDataModule(**asdict(config.datamodule_config))
    module = ImageModule(**asdict(config.module_config), config=asdict(config))

    logger = WandbLogger(
        name=config.logger_config.experiment_name,
        project=config.logger_config.project,
    )

    profiler = AdvancedProfiler()

    callbacks = []

    callbacks.append(ModelCheckpoint(
        monitor=config.callbacks_config.checkpoint_monitor,
        mode=config.callbacks_config.checkpoint_mode,
        save_top_k=config.callbacks_config.checkpoint_save_top_k,
    ))

    callbacks.append(EarlyStopping(
        monitor=config.callbacks_config.early_stopping_monitor,
        mode=config.callbacks_config.early_stopping_mode,
        patience=config.callbacks_config.early_stopping_patience,
    ))

    if config.callbacks_config.learning_rate_monitor:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        accelerator='gpu',
        devices=[config.device],
        default_root_dir=config.default_root_dir,
        profiler=profiler,
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )

    return datamodule, module, trainer

if __name__ == '__main__':
    config = parse_args()
    datamodule, module, trainer = setup_pretraining(config)
    trainer.fit(module, datamodule=datamodule)
    # best_model_path = trainer.checkpoint_callback.best_model_path
    # best_model = ImageModule.load_from_checkpoint(best_model_path, config=asdict(config))
    # trainer.test(best_model, datamodule=datamodule)

