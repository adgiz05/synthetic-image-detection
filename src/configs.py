from dataclasses import dataclass, field, asdict

@dataclass
class PretrainingDatasetConfig:
    task            : str = 'all'
    augmentation    : str = 'patched'
    size            : int = 96
    n_views         : int = 1
    randaug         : bool = True

@dataclass
class PretrainingDataModuleConfig:
    batch_size      : int = 200
    num_workers     : int = 4
    dataset_config  : PretrainingDatasetConfig = field(default_factory=PretrainingDatasetConfig)

@dataclass
class PretrainingOptimizerConfig:
    name            : str = 'sgd'
    lr              : float = 5e-3
    weight_decay    : float = 1e-2

@dataclass
class PretrainingSchedulerConfig:
    name            : str = 'linear+cosine'
    max_epochs      : int = 400
    scheduler_skip  : int = 10

@dataclass
class PretrainingModuleConfig:
    model           : str = 'conresnet'
    optimizer_config: PretrainingOptimizerConfig = field(default_factory=PretrainingOptimizerConfig)
    scheduler_config: PretrainingSchedulerConfig = field(default_factory=PretrainingSchedulerConfig)

@dataclass
class PretrainingLoggerConfig:
    name            : str = 'wandb'
    project         : str = 'imaginet-pretraining'
    experiment_name : str = 'conresnet-pretraining'

@dataclass
class PretrainingCallbacksConfig:
    checkpoint_monitor      : str = 'val/loss'
    checkpoint_mode         : str = 'min'
    checkpoint_save_top_k   : int = 1
    early_stopping_monitor  : str = 'val/loss'
    early_stopping_mode     : str = 'min'
    early_stopping_patience : int = 50
    learning_rate_monitor   : bool = True

@dataclass
class PretrainingConfig:
    datamodule_config           : PretrainingDataModuleConfig = field(default_factory=PretrainingDataModuleConfig)
    module_config               : PretrainingModuleConfig = field(default_factory=PretrainingModuleConfig)
    logger_config               : PretrainingLoggerConfig = field(default_factory=PretrainingLoggerConfig)
    callbacks_config            : PretrainingCallbacksConfig = field(default_factory=PretrainingCallbacksConfig)
    max_epochs                  : int = 400
    precision                   : str = 'bf16-mixed'
    device                      : int = 0
    default_root_dir            : str = 'results/pretraining'
    check_val_every_n_epochs    : int = 5
    accumulate_grad_batches     : int = 1