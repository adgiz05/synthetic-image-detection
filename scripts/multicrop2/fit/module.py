"""
PyTorch Lightning Module for SelfCon Encoder training.

This module encapsulates the complete training logic for the SelfCon encoder,
including model, loss, optimizer, and scheduler configuration.
"""
import torch
import torch.nn as nn
import lightning as L
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import List, Optional, Tuple

from loss import MultiLabelLoss
from grad_cache import GradCache
from models import ConResNet


class SelfConEncoderModule(L.LightningModule):
    """
    PyTorch Lightning Module for SelfCon Encoder training.

    This module replaces the manual training loop from phase1_encoder.py
    with a clean Lightning implementation.

    Args:
        arch: Model architecture name (e.g., "resnet50nodown")
        selfcon_pos: List of booleans indicating which layers have SelfCon
        selfcon_arch: Sub-network architecture type
        selfcon_size: Sub-network size type ("fc", "same", "small", "large")
        feat_dim: Feature dimension for projection head
        pretrained_weights: Source of pretrained weights ("imagenet", "dino", "none")
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        momentum: Momentum for SGD
        optimizer_name: Optimizer type ("sgd", "adam", "adamw", "lars")
        temp: Temperature for contrastive loss
        epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        cosine: Whether to use cosine annealing scheduler
        grad_cache: Whether to use gradient caching
        grad_cache_chunk_size: Chunk size for gradient caching

    Note:
        When grad_cache is enabled, we use manual optimization because GradCache
        handles backward passes internally.
    """

    def __init__(
        self,
        # Model config
        arch: str = "resnet50nodown",
        selfcon_pos: List[bool] = [False, True, False],
        selfcon_arch: str = "resnet",
        selfcon_size: str = "fc",
        feat_dim: int = 128,
        pretrained_weights: str = "imagenet",
        # BayarConv2d config (optional residue extraction layer)
        use_bayar_conv: bool = False,
        bayar_kernel_size: int = 5,
        # DT-CWT config (optional wavelet front-end)
        use_dtcwt: bool = False,
        # Modality dropout (zero-out RGB to force wavelet learning)
        modality_dropout_p: float = 0.0,
        # SE-Block config (channel recalibration after stem conv)
        use_se_block: bool = False,
        se_reduction: int = 16,
        # Optimizer config
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        momentum: float = 0.9,
        optimizer_name: str = "sgd",
        # Loss config
        temp: float = 0.07,
        hierarchical_loss: bool = False,
        inter_family_weight: float = 0.5,
        binary_loss: bool = False,
        # Scheduler config
        epochs: int = 100,
        warmup_epochs: int = 10,
        cosine: bool = True,
        # GradCache config
        grad_cache: bool = True,
        grad_cache_chunk_size: int = 80,
        # Multi-crop config
        num_crops: int = 1,
        # Invariance loss config (Option B)
        invariance_weight: float = 0.0,
        invariance_detach_clean: bool = True,
        invariance_contrastive_degraded: bool = False,
        invariance_warmup_epochs: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build encoder model
        self.model = ConResNet(
            name=arch,
            selfcon_pos=selfcon_pos,
            selfcon_arch=selfcon_arch,
            selfcon_size=selfcon_size,
            feat_dim=feat_dim,
            dataset="imaginet",
            use_bayar_conv=use_bayar_conv,
            bayar_kernel_size=bayar_kernel_size,
            use_dtcwt=use_dtcwt,
            modality_dropout_p=modality_dropout_p,
            use_se_block=use_se_block,
            se_reduction=se_reduction,
        )

        # Load pretrained weights if requested
        # Note: When use_bayar_conv=True, weights for conv1 are incompatible
        # Note: When use_dtcwt=True, conv1 has 21 input channels — partial load needed
        if pretrained_weights != "none":
            self._load_pretrained_weights(
                source=pretrained_weights,
                skip_conv1=use_bayar_conv,
                partial_conv1=use_dtcwt,
            )

        # Build loss function
        self.criterion = MultiLabelLossImagiNet(
            temp_1=temp, temp_base_1=temp,
            temp_2=temp, temp_base_2=temp,
            temp_3=temp, temp_base_3=temp,
        )

        # GradCache wrapper (if enabled)
        self.use_grad_cache = grad_cache
        self._grad_cache = None  # Lazy initialization

        # Use manual optimization when GradCache is enabled
        # because GradCache handles backward passes internally
        self.automatic_optimization = not grad_cache

    def _init_grad_cache(self):
        """Initialize GradCache (called after model is on device)."""
        if self._grad_cache is None and self.use_grad_cache:
            is_distributed = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )
            self._grad_cache = GradCache(
                model=self.model,
                chunk_size=self.hparams.grad_cache_chunk_size,
                loss_fn=self.criterion,
                loss_type="SelfCon",
                distributed=is_distributed,
                lightning_module=self.trainer.model if is_distributed else None,
            )

    def _load_pretrained_weights(
        self,
        source: str = "imagenet",
        skip_conv1: bool = False,
        partial_conv1: bool = False,
    ):
        """Load pretrained weights into the encoder.

        Args:
            source: Weight source — "imagenet", "dino", "dinov3", or a local .pth path.
            skip_conv1: If True, skip loading conv1 weights (useful when using BayarConv2d
                       as the first layer, since the input channels differ).
            partial_conv1: If True, load only the first 3 channels of conv1 weights from
                          pretrained and initialize remaining channels with Kaiming
                          (useful when using DT-CWT, which adds 18 extra input channels).
        """
        arch = self.hparams.arch
        is_convnext = arch.startswith("convnext_")

        if is_convnext:
            self._load_convnext_pretrained_weights(source, arch)
        else:
            self._load_resnet_pretrained_weights(source, arch, skip_conv1, partial_conv1)

    def _load_resnet_pretrained_weights(
        self, source: str, arch: str, skip_conv1: bool, partial_conv1: bool,
    ):
        """Load pretrained weights for ResNet-family encoders."""
        resnet_dict, pretrained_name = self._get_resnet_pretrained_state_dict(source, arch)

        encoder_dict = self.model.encoder.state_dict()

        # Map pretrained weights to encoder
        loaded_count = 0
        skipped_count = 0
        partial_count = 0
        for name, param in encoder_dict.items():
            # Skip SelfCon-specific layers
            if "selfcon" in name:
                continue

            # Skip BayarConv2d layers (they have their own initialization)
            if "bayar" in name:
                continue

            # Skip DT-CWT frontend layers (BatchNorm is trained from scratch)
            if "dtcwt_frontend" in name:
                continue

            # Skip conv1 when using BayarConv2d (input channels differ)
            if skip_conv1 and "conv1" in name:
                skipped_count += 1
                continue

            # Partial conv1 load for DT-CWT: copy RGB channels, Kaiming-init the rest
            if partial_conv1 and name == "conv1.weight":
                source_key = name.replace("shortcut", "downsample")
                if source_key in resnet_dict:
                    pretrained_weight = resnet_dict[source_key]  # (64, 3, 7, 7)
                    # Copy first 3 channels (RGB) from pretrained
                    param[:, :3, :, :].copy_(pretrained_weight)
                    # Initialize remaining 18 channels (DT-CWT) with Kaiming
                    nn.init.kaiming_normal_(param[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
                    partial_count += 1
                continue

            # Convert naming convention (shortcut -> downsample)
            source_key = name.replace("shortcut", "downsample")

            if source_key in resnet_dict and resnet_dict[source_key].shape == param.shape:
                param.copy_(resnet_dict[source_key])
                loaded_count += 1

        msg = f"Loaded {loaded_count} pretrained weights from {pretrained_name}"
        if skipped_count > 0:
            msg += f" (skipped {skipped_count} conv1 weights due to BayarConv2d)"
        if partial_count > 0:
            msg += f" (partially loaded {partial_count} conv1 weights for DT-CWT: 3ch pretrained + 18ch Kaiming)"
        print(msg)

    def _load_convnext_pretrained_weights(self, source: str, arch: str):
        """Load pretrained weights for ConvNeXt encoders.

        Supports:
            - "imagenet": torchvision ImageNet-supervised weights
            - "dinov3": DINOv3 distilled ConvNeXt weights (from local .pth file)
            - A local file path ending in .pth/.pt: load directly from file
        """
        pretrained_dict, pretrained_name = self._get_convnext_pretrained_state_dict(source, arch)

        encoder_dict = self.model.encoder.state_dict()

        loaded_count = 0
        for name, param in encoder_dict.items():
            # Skip SelfCon-specific layers (trained from scratch)
            if "selfcon" in name:
                continue
            if name in pretrained_dict and pretrained_dict[name].shape == param.shape:
                param.copy_(pretrained_dict[name])
                loaded_count += 1

        print(f"Loaded {loaded_count}/{len(encoder_dict)} pretrained weights from {pretrained_name}")

    @staticmethod
    def _get_resnet_pretrained_state_dict(source: str, arch: str):
        """Get pretrained state dict for ResNet-family from the specified source.

        Returns:
            Tuple of (state_dict, pretrained_name).
        """
        if source == "imagenet":
            try:
                if 'resnext101' in arch:
                    from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
                    pretrained_model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
                    pretrained_name = "ImageNet ResNeXt101-32x8d"
                else:
                    from torchvision.models import resnet50, ResNet50_Weights
                    pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                    pretrained_name = "ImageNet ResNet50"
            except ImportError:
                if 'resnext101' in arch:
                    from torchvision.models import resnext101_32x8d
                    pretrained_model = resnext101_32x8d(pretrained=True)
                    pretrained_name = "ImageNet ResNeXt101-32x8d"
                else:
                    from torchvision.models import resnet50
                    pretrained_model = resnet50(pretrained=True)
                    pretrained_name = "ImageNet ResNet50"
            return pretrained_model.state_dict(), pretrained_name

        elif source == "dino":
            if 'resnext101' in arch:
                raise ValueError("DINO pretrained weights are only available for ResNet50, not ResNeXt101.")
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
                map_location="cpu",
            )
            return state_dict, "DINO ResNet50"

        else:
            raise ValueError(
                f"Unknown pretrained weights source for ResNet: {source}. "
                "Use 'imagenet', 'dino', or 'none'."
            )

    @staticmethod
    def _get_convnext_pretrained_state_dict(source: str, arch: str):
        """Get pretrained state dict for ConvNeXt from the specified source.

        Our ConvNeXtEncoder uses timm's naming convention (stem, stages.N),
        so DINOv3/timm weights load directly. For ImageNet we use timm too.

        Returns:
            Tuple of (state_dict, pretrained_name).
        """
        variant = arch.replace("convnext_", "")  # e.g. "base"

        if source == "imagenet":
            # Use timm to get ImageNet-pretrained ConvNeXt weights
            try:
                import timm
                model = timm.create_model(f'convnext_{variant}', pretrained=True, num_classes=0)
                return model.state_dict(), f"ImageNet ConvNeXt-{variant.capitalize()} (timm)"
            except ImportError:
                # Fallback: torchvision with key remapping
                return _load_torchvision_convnext_weights(variant)

        elif source == "dinov3" or source.endswith(('.pth', '.pt')):
            # Load from local file (DINOv3 weights in timm format)
            if source == "dinov3":
                import os
                weights_dir = os.environ.get(
                    "DINOV3_WEIGHTS_DIR",
                    os.path.expanduser("~/.cache/dinov3"),
                )
                weight_path = os.path.join(weights_dir, f"dinov3_convnext_{variant}.pth")
                if not os.path.exists(weight_path):
                    raise FileNotFoundError(
                        f"DINOv3 weights not found at {weight_path}. "
                        f"Download them and place at this path, or set DINOV3_WEIGHTS_DIR env var. "
                        f"Alternatively, pass the full path as pretrained_weights."
                    )
                pretrained_name = f"DINOv3 ConvNeXt-{variant.capitalize()}"
            else:
                weight_path = source
                pretrained_name = f"Local weights ({source})"

            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)

            # Handle different checkpoint wrappers
            if isinstance(checkpoint, dict):
                for key in ["model", "state_dict", "teacher", "student"]:
                    if key in checkpoint:
                        checkpoint = checkpoint[key]
                        break

            # Detect key format and strip prefixes if needed
            sample_key = next(iter(checkpoint.keys()), "")
            if sample_key.startswith("backbone."):
                checkpoint = {k.replace("backbone.", "", 1): v for k, v in checkpoint.items()}
            elif sample_key.startswith("encoder."):
                checkpoint = {k.replace("encoder.", "", 1): v for k, v in checkpoint.items()}

            # timm DINOv3 weights should already match (stem.*, stages.*)
            return checkpoint, pretrained_name

        else:
            raise ValueError(
                f"Unknown pretrained weights source for ConvNeXt: {source}. "
                "Use 'imagenet', 'dinov3', a local .pth path, or 'none'."
            )


    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
        """Training step."""
        images, labels = batch
        K = self.hparams.num_crops

        if K > 1:
            # images: (B, K, C, H, W) → (B*K, C, H, W)
            B = images.shape[0]
            images = images.view(B * K, *images.shape[2:])

        if self.use_grad_cache:
            if self.hparams.invariance_weight > 0:
                raise RuntimeError(
                    "GradCache does not support invariance loss. "
                    "Set grad_cache: false when using invariance_weight > 0."
                )
            # Manual optimization mode for GradCache
            opt = self.optimizers()

            # Zero gradients
            opt.zero_grad()

            # Initialize GradCache if needed (lazy init to ensure model is on device)
            self._init_grad_cache()

            # GradCache handles forward and backward internally
            loss = self._grad_cache(images, labels, num_crops=K)

            # Step optimizer (gradients already accumulated by GradCache)
            opt.step()

            # Log metrics
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/lr", opt.param_groups[0]['lr'], on_step=True)

            # Return None (manual optimization)
            return None
        else:
            # Automatic optimization mode
            loss = self._compute_selfcon_loss(images, labels, num_crops=K)

            # Log metrics
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)

            return loss

    def _compute_selfcon_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        num_crops: int = 1,
    ) -> torch.Tensor:
        """Compute SelfCon loss, optionally with invariance regularization."""
        use_invariance = self.hparams.invariance_weight > 0

        if use_invariance:
            # num_crops must be even: K clean + K degraded interleaved
            assert num_crops % 2 == 0, (
                f"invariance_weight > 0 requires even num_crops (got {num_crops}). "
                "Configure degradation_view_transforms in data."
            )
            K = num_crops // 2  # actual crop locations (1 for B1C, 3 for multicrop+B1C)

            # images arrives as (B*num_crops, C, H, W) after view() in training_step
            # Interleaved: [img0_clean, img0_deg, img1_clean, img1_deg, ...]
            clean_images = images[0::2]      # (B*K, C, H, W)
            degraded_images = images[1::2]   # (B*K, C, H, W)
            B = clean_images.shape[0] // K if K > 1 else clean_images.shape[0]

            # Forward clean views (used for contrastive + invariance target)
            f1_clean, f2_clean, enc_clean = self.model(clean_images, return_encoder_features=True)

            # Forward degraded views
            if self.hparams.invariance_contrastive_degraded:
                # Full forward: degraded features also participate in contrastive loss
                f1_deg, f2_deg, enc_degraded = self.model(degraded_images, return_encoder_features=True)

                # Combine features: (B*K, n_views, feat_dim)
                features_clean = torch.cat(
                    [f.unsqueeze(1) for f in f1_clean] + [f2_clean.unsqueeze(1)], dim=1
                )
                features_deg = torch.cat(
                    [f.unsqueeze(1) for f in f1_deg] + [f2_deg.unsqueeze(1)], dim=1
                )

                # Reshape for multicrop: (B*K, n_views, feat) → (B, K*n_views, feat)
                if K > 1:
                    n_views = features_clean.shape[1]
                    features_clean = features_clean.view(B, K * n_views, -1)
                    features_deg = features_deg.view(B, K * n_views, -1)

                features = torch.cat([features_clean, features_deg], dim=1)
            else:
                # Original B1: degraded only used for MSE invariance
                _, _, enc_degraded = self.model(degraded_images, return_encoder_features=True)

                features = torch.cat(
                    [f.unsqueeze(1) for f in f1_clean] + [f2_clean.unsqueeze(1)], dim=1
                )
                if K > 1:
                    n_views = features.shape[1]
                    features = features.view(B, K * n_views, -1)

            loss_contrastive = self.criterion(features, labels)

            # Invariance loss: MSE between clean and degraded encoder features (2048-dim)
            # Operates on all B*K pairs directly (no reshape needed)
            if self.hparams.invariance_detach_clean:
                target = enc_clean.detach()
            else:
                target = enc_clean
            loss_invariance = nn.functional.mse_loss(
                nn.functional.normalize(enc_degraded, dim=1),
                nn.functional.normalize(target, dim=1),
            )

            # Warmup schedule (curriculum λ)
            warmup = self.hparams.invariance_warmup_epochs
            if warmup > 0 and self.current_epoch < warmup:
                lambda_t = self.hparams.invariance_weight * (self.current_epoch / warmup)
            else:
                lambda_t = self.hparams.invariance_weight

            # Monitor collapse: mean std across encoder dimensions (should stay > 0)
            enc_std = enc_degraded.std(dim=0).mean()

            # Log components
            self.log("train/loss_contrastive", loss_contrastive, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/loss_invariance", loss_invariance, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/enc_std_degraded", enc_std, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train/invariance_lambda", lambda_t, on_step=False, on_epoch=True, sync_dist=True)

            loss = loss_contrastive + lambda_t * loss_invariance
        else:
            f1, f2 = self.model(images)

            # Combine features from sub-networks and main network
            # features: (B*K, n_views, feat_dim)
            features = torch.cat([f.unsqueeze(1) for f in f1] + [f2.unsqueeze(1)], dim=1)

            if num_crops > 1:
                # Reshape: (B*K, n_views, feat_dim) → (B, K*n_views, feat_dim)
                BK, n_views, feat_dim = features.shape
                B = BK // num_crops
                features = features.view(B, num_crops * n_views, feat_dim)

            loss = self.criterion(features, labels)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Select optimizer
        if self.hparams.optimizer_name == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer_name == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer_name == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer_name == "lars":
            # Separate BN/bias params (no LARS scaling, no weight decay)
            bn_bias_params = []
            other_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "bn" in name or "bias" in name:
                    bn_bias_params.append(param)
                else:
                    other_params.append(param)
            optimizer = LARS(
                [
                    {"params": other_params},
                    {"params": bn_bias_params, "weight_decay": 0, "exclude_from_lars": True},
                ],
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer_name}")

        # Build scheduler list
        schedulers = []

        # Warmup scheduler
        if self.hparams.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=self.hparams.warmup_epochs
            )
            schedulers.append(warmup_scheduler)

        # Cosine annealing scheduler
        if self.hparams.cosine:
            remaining_epochs = self.hparams.epochs - self.hparams.warmup_epochs
            eta_min = self.hparams.learning_rate * (0.1 ** 3)
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=remaining_epochs,
                eta_min=eta_min
            )
            schedulers.append(cosine_scheduler)

        # Combine schedulers
        if len(schedulers) > 1:
            scheduler = SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=[self.hparams.warmup_epochs]
            )
        elif len(schedulers) == 1:
            scheduler = schedulers[0]
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Manually step scheduler when using manual optimization (GradCache mode)
        if self.use_grad_cache:
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()

    def on_save_checkpoint(self, checkpoint):
        """Called when saving a checkpoint."""
        # Save hyperparameters for easy loading
        checkpoint['hyper_parameters'] = dict(self.hparams)

    def on_load_checkpoint(self, checkpoint):
        """Called when loading a checkpoint."""
        pass


def _load_torchvision_convnext_weights(variant: str):
    """Fallback: load torchvision ImageNet weights and remap keys to timm naming."""
    from torchvision.models import (
        convnext_tiny, convnext_small, convnext_base, convnext_large,
        ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights,
        ConvNeXt_Base_Weights, ConvNeXt_Large_Weights,
    )
    builders = {
        'tiny': (convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        'small': (convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1),
        'base': (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1),
        'large': (convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1),
    }
    builder_fn, weights = builders[variant]
    pretrained_model = builder_fn(weights=weights)
    tv_state = pretrained_model.state_dict()

    # Remap torchvision keys (features.N.) to timm keys (stem., stages.N.)
    # torchvision: features.0=stem, features.1=stage0_blocks,
    #              features.2=down01, features.3=stage1_blocks, ...
    # timm: stem, stages.0 (downsample+blocks), stages.1, ...
    tv_to_timm = [
        ('features.0.', 'stem.'),
        ('features.1.', 'stages.0.blocks.'),
        ('features.2.', 'stages.1.downsample.'),
        ('features.3.', 'stages.1.blocks.'),
        ('features.4.', 'stages.2.downsample.'),
        ('features.5.', 'stages.2.blocks.'),
        ('features.6.', 'stages.3.downsample.'),
        ('features.7.', 'stages.3.blocks.'),
    ]
    remapped = {}
    for key, value in tv_state.items():
        new_key = key
        for tv_prefix, timm_prefix in tv_to_timm:
            if key.startswith(tv_prefix):
                new_key = key.replace(tv_prefix, timm_prefix, 1)
                break
        # Skip classifier head keys
        if new_key.startswith('classifier.'):
            continue
        remapped[new_key] = value
    return remapped, f"ImageNet ConvNeXt-{variant.capitalize()} (torchvision)"
