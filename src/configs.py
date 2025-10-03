from dataclasses import dataclass, field

# Synthetic CLS
@dataclass
class AugmentationConfig:
    transformations: bool = True
    downscaling_prob: float = 0.3
    resize_prob: float = 0.3
    compression_prob: float = 0.3
    blur_noise_prob: float = 0.1
    color_prob: float = 0.1
    texture_prob: float = 0.1
    local_artifacts_prob: float = 0.1
    rotation_prob: float = 0.1
    flip_prob: float = 0.1

@dataclass
class ImageDatasetConfig:
    task                : str = 'all'
    size                : int = 224
    augmentation        : str = 'patched'
    augmentation_config : AugmentationConfig = field(default_factory=AugmentationConfig)
    dataset_size        : str = 'full'  # 'full' or 'reduced'

@dataclass
class ImageDataModuleConfig:
    batch_size              : int = 128
    num_workers             : int = 8
    train_dataset_config    : ImageDatasetConfig = field(default_factory=ImageDatasetConfig)
    val_dataset_config      : ImageDatasetConfig = field(default_factory=ImageDatasetConfig)

@dataclass
class ImageOptimizerConfig:
    name            : str = 'adamw'
    lr              : float = 5e-4
    weight_decay    : float = 1e-2

@dataclass
class ImageSchedulerConfig:
    name            : str = 'linear+cosine'
    max_epochs      : int = 400
    scheduler_skip  : int = 10

@dataclass
class ImageLossConfig:
    synthetic_weight  : float = 1.0
    model_weight      : float = 1.0
    model_loss        : str = 'generator'  # 'generator' or 'specific_model'

@dataclass
class ImageModuleConfig:
    model_id            : str = 'google/vit-base-patch16-224-in21k'
    optimizer_config    : ImageOptimizerConfig = field(default_factory=ImageOptimizerConfig)
    scheduler_config    : ImageSchedulerConfig = field(default_factory=ImageSchedulerConfig)
    loss_config         : ImageLossConfig = field(default_factory=ImageLossConfig)

@dataclass
class ImageLoggerConfig:
    name            : str = 'wandb'
    project         : str = 'imaginet-cls'
    experiment_name : str = 'baseline'

@dataclass
class ImageCallbacksConfig:
    checkpoint_monitor      : str = 'val/loss'
    checkpoint_mode         : str = 'min'
    checkpoint_save_top_k   : int = 1
    early_stopping_monitor  : str = 'val/loss'
    early_stopping_mode     : str = 'min'
    early_stopping_patience : int = 50
    learning_rate_monitor   : bool = True

@dataclass
class ImageConfig:
    datamodule_config           : ImageDataModuleConfig = field(default_factory=ImageDataModuleConfig)
    module_config               : ImageModuleConfig = field(default_factory=ImageModuleConfig)
    logger_config               : ImageLoggerConfig = field(default_factory=ImageLoggerConfig)
    callbacks_config            : ImageCallbacksConfig = field(default_factory=ImageCallbacksConfig)
    max_epochs                  : int = 400
    precision                   : str = 'bf16-mixed'
    device                      : int = 0
    default_root_dir            : str = 'results/pretraining'
    check_val_every_n_epoch     : int = 5
    accumulate_grad_batches     : int = 1

# Pretraining configs
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
    check_val_every_n_epoch     : int = 5
    accumulate_grad_batches     : int = 1