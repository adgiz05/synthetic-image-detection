from src.datamodules import SelfContrastivePretrainingDataModule
from src.modules import SelfContrastivePretrainingModule
from src.configs import *

import os

import pytorch_lightning as pl
import argparse

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

def parse_args():
    parser = argparse.ArgumentParser(description='Pretraining')
    parser.add_argument('--task', type=str, default='all', help='Loss task')
    parser.add_argument('--size', type=int, default=96, help='Image size')
    parser.add_argument('--n_views', type=int, default=1, help='Views per image')
    parser.add_argument('--randaug', action='store_true', help='Use RandAugment')

    parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer (sgd, adam, adamw)')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')

    parser.add_argument('--scheduler', type=str, default='linear+cosine', help='Scheduler (linear, cosine, linear+cosine)')
    parser.add_argument('--scheduler_skip', type=int, default=10, help='Number of epochs to skip from linear to cosine')
    
    parser.add_argument('--model', type=str, default='conresnet', help='Backbone model')

    parser.add_argument('--project', type=str, default='imaginet-pretraining', help='Project name')
    parser.add_argument('--experiment', type=str, default='conresnet-pretraining', help='Experiment name')

    parser.add_argument('--early_stopping_patience', type=int, default=50, help='Early stopping patience')

    parser.add_argument('--max_epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--precision', type=str, default='bf16-mixed', help='Precision (32, 16, bf16-mixed)')
    parser.add_argument('--device', type=int, default=0, help='GPU to train in')
    parser.add_argument('--check_val_every_n_epochs', type=int, default=5, help='Check val every n epochs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulate grad batches')

    args = parser.parse_args()

    dataset_config = PretrainingDatasetConfig(task=args.task, size=args.size, n_views=args.n_views, randaug=args.randaug)
    datamodule_config = PretrainingDataModuleConfig(batch_size=args.batch_size, num_workers=args.num_workers, dataset_config=dataset_config)
    optimizer_config = PretrainingOptimizerConfig(name=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
    scheduler_config = PretrainingSchedulerConfig(name=args.scheduler, max_epochs=args.max_epochs, scheduler_skip=args.scheduler_skip)
    module_config = PretrainingModuleConfig(model=args.model, optimizer_config=optimizer_config, scheduler_config=scheduler_config)
    logger_config = PretrainingLoggerConfig(project=args.project, experiment_name=args.experiment)
    callbacks_config = PretrainingCallbacksConfig(early_stopping_patience=args.early_stopping_patience)
    
    return PretrainingConfig(
        datamodule_config=datamodule_config,
        module_config=module_config,
        logger_config=logger_config,
        callbacks_config=callbacks_config,
        max_epochs=args.max_epochs,
        precision=args.precision,
        device=args.device,
        default_root_dir='results/pretraining',
        check_val_every_n_epochs=args.check_val_every_n_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

def setup_pretraining(config):
