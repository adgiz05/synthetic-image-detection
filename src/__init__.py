"""
Synthetic Image Detection Library
Shared components for training and testing full-image classifiers.
"""

__version__ = "0.1.0"

# Export main components for easy import
from .constants import IMAGENET_MEAN, IMAGENET_STD, PROJECT
from .models import FullImageModule, AttnAggregator
from .losses import Phase1Loss
from .tube_model import (
    PatchEncoder,
    FusionMLP,
    ProjectionHead,
    LocalScoreHead,
    TubeModel,
    TubeContrastiveModule,
)
from .datasets import FullImageDataset
from .collators import TrainFullImageDegradePatchCollator, ValFullImagePatchCollator
from .metrics import compute_binary_auc, safe_prf
from .patch_utils import extract_patches_tensor_auto, sliding_window_indices

__all__ = [
    # Constants
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PROJECT",
    # Models
    "FullImageModule",
    "AttnAggregator",
    # Losses
    "Phase1Loss",
    # Tube model
    "PatchEncoder",
    "FusionMLP",
    "ProjectionHead",
    "LocalScoreHead",
    "TubeModel",
    "TubeContrastiveModule",
    # Datasets
    "FullImageDataset",
    # Collators
    "TrainFullImageDegradePatchCollator",
    "ValFullImagePatchCollator",
    # Metrics
    "compute_binary_auc",
    "safe_prf",
    # Patch utilities
    "extract_patches_tensor_auto",
    "sliding_window_indices",
]
