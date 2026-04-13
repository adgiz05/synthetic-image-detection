"""
Tube-based forensics model for multi-scale contrastive learning.

Architecture (per patch):
  tubes         [B, N, K, V, 6, P, P]  →  enc_spatial  (6-ch ResNet-18)  →  h_sp  [D_enc]
  tubes_wavelet [B, N, K, V, 3, P, P]  →  enc_wavelet  (3-ch ResNet-18)  →  h_fr  [D_enc]
                                                              ↓
                                                       FusionMLP  →  h  [D_fused]
                                                       /               \\
                                               proj_auth               proj_src
                                                   ↓                       ↓
                                             z_auth [D_auth]          z_src [D_src]
                                             (L2-norm)                (L2-norm)
                                                   ↓
                                             local_score → a_i (scalar)

Scale aggregation (mean over K):
  h_tube [B, N, D_fused],  z_auth_tube [B, N, D_auth],  z_src_tube [B, N, D_src]

MIL aggregation (attention pooling over N tubes):
  h_img [B, D_fused]

Classification heads:
  binary_head(h_img) → logits_auth [B, 2]
  source_head(h_img) → logits_src  [B, M]   (optional)

# Contrastive / decoupling losses to be added in a later step.
"""

import math
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import pytorch_lightning as pl

from .constants import IMAGENET_MEAN, IMAGENET_STD
from .metrics import compute_binary_auc
from .models import AttnAggregator
from .losses import Phase1Loss
from .gpu_preprocessing import GPUPreprocessor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    """
    ResNet-18 based encoder that accepts an arbitrary number of input channels.

    For the spatial branch: in_channels=6 (3 RGB + 3 residual).
    For the wavelet branch: in_channels=3 (LH / HL / HH).

    When pretrained=True and in_channels>3, the RGB channels of the first
    conv are initialized from ImageNet weights; any extra channels are
    zero-initialized (so they start neutral and the network learns to
    exploit them without corrupting the pretrained filters).

    Args:
        in_channels:    number of input channels
        out_dim:        output embedding dim (projects from ResNet's 512)
        pretrained:     load ImageNet weights for the backbone
    """

    def __init__(self, in_channels: int, out_dim: int = 256, pretrained: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Load ResNet-18 (version-safe: newer torchvision uses `weights=` API)
        try:
            weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
            base = tvm.resnet18(weights=weights)
        except AttributeError:          # torchvision < 0.13
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base = tvm.resnet18(pretrained=pretrained)

        # Adapt first conv when in_channels != 3
        if in_channels != 3:
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:, :3] = base.conv1.weight.data   # copy RGB filters
                    if in_channels > 3:
                        nn.init.zeros_(new_conv.weight[:, 3:])         # zero extra channels
            base.conv1 = new_conv

        # Drop final FC; keep everything up to the global average pool
        # children order: conv1, bn1, relu, maxpool, layer1..4, avgpool, fc
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # → [B, 512, 1, 1]

        self.proj = nn.Linear(512, out_dim) if out_dim != 512 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, H, W]  →  [B, out_dim]"""
        h = self.backbone(x).flatten(1)  # [B, 512]
        return self.proj(h)              # [B, out_dim]


class FusionMLP(nn.Module):
    """
    Merge spatial and frequency embeddings:
      h = MLP( cat(h_sp, h_fr) )

    Two linear layers with GELU activation and LayerNorm in between for
    stable training when both branches learn at different rates.

    Args:
        dim_sp:     spatial encoder output dim
        dim_fr:     wavelet encoder output dim
        fused_dim:  output dimension
    """

    def __init__(self, dim_sp: int, dim_fr: int, fused_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_sp + dim_fr, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, h_sp: torch.Tensor, h_fr: torch.Tensor) -> torch.Tensor:
        """h_sp / h_fr: [*, dim_sp/fr]  →  [*, fused_dim]"""
        return self.net(torch.cat([h_sp, h_fr], dim=-1))


class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head for contrastive learning.
    Output lives on the unit hypersphere (L2-normalized).

    Args:
        in_dim:     input dimension
        hidden_dim: hidden layer width
        out_dim:    output (embedding) dimension
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [*, in_dim]  →  [*, out_dim]  (L2 normalized)"""
        return F.normalize(self.net(x), dim=-1)


class LocalScoreHead(nn.Module):
    """
    Per-patch/tube authenticity scoring head.
    Produces a raw logit (unbounded) used for MIL / top-k patch selection.

    Args:
        in_dim: input dimension (operates on z_auth)
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [*, in_dim]  →  [*, 1]  raw logit"""
        return self.net(z)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class TubeModel(nn.Module):
    """
    Core forensics model for multi-scale tube learning.

    Takes the output of MultiScaleTubeCollator or FastMultiScaleTubeCollator and produces:
      - z_auth:       [B, N, D_auth]   per-tube auth embeddings (contrastive)
      - z_src:        [B, N, D_src]    per-tube source embeddings (contrastive)
      - a_local:      [B, N]           per-tube local auth score (MIL / top-k)
      - h_img:        [B, D_fused]     image-level representation (classification)
      - attn_weights: [B, N]           MIL attention weights
      - logits_auth:  [B, 2]           binary classification logits
      - logits_src:   [B, M] or None   source classification logits

    Optimizations:
      - GPU preprocessing: residual/wavelet computed on GPU (not in collator)
      - Batched view encoding: all views processed in single forward pass

    Args:
        encoder_dim:        output dim of each PatchEncoder branch
        fused_dim:          dim after FusionMLP
        z_auth_dim:         auth projection space dim
        z_src_dim:          src  projection space dim
        attn_dim:           dim for MIL attention MLP
        num_src_classes:    number of generator classes (None → no source head)
        pretrained_spatial: use ImageNet weights for the spatial (6-ch) encoder
        target_size:        patch size for GPU preprocessing
    """

    def __init__(
        self,
        encoder_dim: int = 256,
        fused_dim: int = 256,
        z_auth_dim: int = 128,
        z_src_dim: int = 128,
        attn_dim: int = 128,
        num_src_classes: Optional[int] = None,
        pretrained_spatial: bool = False,
        target_size: int = 96,
    ):
        super().__init__()
        self.fused_dim = fused_dim
        self.z_auth_dim = z_auth_dim
        self.z_src_dim  = z_src_dim
        self.num_src_classes = num_src_classes
        self.target_size = target_size

        # ── GPU Preprocessing ────────────────────────────────────────────────
        self.gpu_preproc = GPUPreprocessor(target_size=target_size)

        # ── ImageNet normalization constants (for GPU) ────────────────────────
        self.register_buffer('img_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        # ── Encoders ──────────────────────────────────────────────────────────
        # Spatial branch: 6 channels = RGB (ImageNet-norm) + high-freq residual
        self.enc_spatial = PatchEncoder(
            in_channels=6, out_dim=encoder_dim, pretrained=pretrained_spatial
        )
        # Frequency branch: 3 channels = Haar DWT LH / HL / HH
        self.enc_wavelet = PatchEncoder(
            in_channels=3, out_dim=encoder_dim, pretrained=False
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = FusionMLP(encoder_dim, encoder_dim, fused_dim)

        # ── Projection heads (for future contrastive losses) ──────────────────
        self.proj_auth = ProjectionHead(fused_dim, fused_dim, z_auth_dim)
        self.proj_src  = ProjectionHead(fused_dim, fused_dim, z_src_dim)

        # ── Local score (MIL) ─────────────────────────────────────────────────
        self.local_score = LocalScoreHead(z_auth_dim)

        # ── MIL aggregation over N tubes ──────────────────────────────────────
        self.mil_agg = AttnAggregator(fused_dim, attn_dim)

        # ── Classification heads ──────────────────────────────────────────────
        self.binary_head = nn.Linear(fused_dim, 2)
        self.source_head = (
            nn.Linear(fused_dim, num_src_classes) if num_src_classes else None
        )

    # -------------------------------------------------------------------------

    def preprocess_rgb(self, rgb: torch.Tensor) -> tuple:
        """
        Preprocess RGB patches to spatial (6-ch) and wavelet (3-ch) inputs.

        Args:
            rgb: [B, 3, H, W] RGB patches in [0, 1] (NOT normalized)

        Returns:
            spatial: [B, 6, H, W] - RGB (ImageNet-norm) + residual
            wavelet: [B, 3, H, H] - Haar DWT bands
        """
        # Compute residual and wavelet on GPU
        residual, wavelet = self.gpu_preproc(rgb)

        # ImageNet normalize RGB
        rgb_norm = (rgb - self.img_mean) / self.img_std

        # Concatenate for spatial branch
        spatial = torch.cat([rgb_norm, residual], dim=1)  # [B, 6, H, W]

        return spatial, wavelet

    def encode_patches(
        self,
        tubes: torch.Tensor,
        tubes_wavelet: torch.Tensor,
        view_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all (B, N, K) patches for a given view index.

        Args:
            tubes:         [B, N, K, V, 6, P, P]
            tubes_wavelet: [B, N, K, V, 3, P, P]
            view_idx:      which view to encode

        Returns:
            h:      [B, N, K, D_fused]   fused patch embeddings
            z_auth: [B, N, K, D_auth]    auth projections (L2-norm)
            z_src:  [B, N, K, D_src]     src  projections (L2-norm)
        """
        B, N, K, V, _, P, _ = tubes.shape

        sp = tubes[:, :, :, view_idx]          # [B, N, K, 6, P, P]
        fr = tubes_wavelet[:, :, :, view_idx]  # [B, N, K, 3, P, P]

        sp_flat = sp.reshape(B * N * K, 6, P, P)
        fr_flat = fr.reshape(B * N * K, 3, P, P)

        h_sp = self.enc_spatial(sp_flat)  # [B*N*K, D_enc]
        h_fr = self.enc_wavelet(fr_flat)  # [B*N*K, D_enc]

        h      = self.fusion(h_sp, h_fr)   # [B*N*K, D_fused]
        z_auth = self.proj_auth(h)          # [B*N*K, D_auth]
        z_src  = self.proj_src(h)           # [B*N*K, D_src]

        return {
            "h":      h.view(B, N, K, self.fused_dim),
            "z_auth": z_auth.view(B, N, K, self.z_auth_dim),
            "z_src":  z_src.view(B, N, K, self.z_src_dim),
        }

    def encode_patches_from_rgb(
        self,
        tubes_rgb: torch.Tensor,
        view_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode patches from RGB input (GPU preprocessing included).

        Args:
            tubes_rgb: [B, N, K, V, 3, P, P] RGB in [0, 1]
            view_idx:  which view to encode

        Returns:
            Same as encode_patches
        """
        B, N, K, V, _, P, _ = tubes_rgb.shape

        rgb = tubes_rgb[:, :, :, view_idx]  # [B, N, K, 3, P, P]
        rgb_flat = rgb.reshape(B * N * K, 3, P, P)

        # GPU preprocessing
        sp_flat, fr_flat = self.preprocess_rgb(rgb_flat)

        h_sp = self.enc_spatial(sp_flat)
        h_fr = self.enc_wavelet(fr_flat)

        h      = self.fusion(h_sp, h_fr)
        z_auth = self.proj_auth(h)
        z_src  = self.proj_src(h)

        return {
            "h":      h.view(B, N, K, self.fused_dim),
            "z_auth": z_auth.view(B, N, K, self.z_auth_dim),
            "z_src":  z_src.view(B, N, K, self.z_src_dim),
        }

    def encode_all_views(
        self,
        tubes: torch.Tensor,
        tubes_wavelet: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all V views (used by contrastive losses).

        Returns:
            z_auth: [V, B, N, D_auth]  scale-averaged auth embeddings per view
            z_src:  [V, B, N, D_src]   scale-averaged src  embeddings per view
        """
        V = tubes.shape[3]
        z_auth_views, z_src_views = [], []

        for v in range(V):
            enc = self.encode_patches(tubes, tubes_wavelet, view_idx=v)
            z_auth_views.append(F.normalize(enc["z_auth"].mean(dim=2), dim=-1))  # [B, N, D_auth]
            z_src_views.append(F.normalize(enc["z_src"].mean(dim=2),  dim=-1))  # [B, N, D_src]

        return {
            "z_auth": torch.stack(z_auth_views, dim=0),  # [V, B, N, D_auth]
            "z_src":  torch.stack(z_src_views,  dim=0),  # [V, B, N, D_src]
        }

    def encode_per_scale_all_views(
        self,
        tubes: torch.Tensor,
        tubes_wavelet: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all V views and K scales without any aggregation.
        Used by Phase 1 contrastive losses that need the full [B, N, K, V, D] tensors.

        OPTIMIZED: Batches all views into a single forward pass instead of V sequential passes.

        Args:
            tubes:         [B, N, K, V, 6, P, P]
            tubes_wavelet: [B, N, K, V, 3, P, P]

        Returns:
            z_auth: [B, N, K, V, D_auth]  per-scale per-view auth embeddings (L2-norm)
            z_src:  [B, N, K, V, D_src]   per-scale per-view src  embeddings (L2-norm)
        """
        B, N, K, V, _, P, _ = tubes.shape

        # Reshape to batch all views together: [B*N*K*V, C, P, P]
        sp_flat = tubes.reshape(B * N * K * V, 6, P, P)
        fr_flat = tubes_wavelet.reshape(B * N * K * V, 3, P, P)

        # Single forward pass for all views
        h_sp = self.enc_spatial(sp_flat)  # [B*N*K*V, D_enc]
        h_fr = self.enc_wavelet(fr_flat)  # [B*N*K*V, D_enc]

        h      = self.fusion(h_sp, h_fr)   # [B*N*K*V, D_fused]
        z_auth = self.proj_auth(h)          # [B*N*K*V, D_auth]
        z_src  = self.proj_src(h)           # [B*N*K*V, D_src]

        # Reshape back to [B, N, K, V, D]
        z_auth = z_auth.view(B, N, K, V, self.z_auth_dim)
        z_src  = z_src.view(B, N, K, V, self.z_src_dim)

        return {"z_auth": z_auth, "z_src": z_src}

    def encode_per_scale_all_views_from_rgb(
        self,
        tubes_rgb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all views from RGB input (GPU preprocessing + batched forward).

        Args:
            tubes_rgb: [B, N, K, V, 3, P, P] RGB in [0, 1]

        Returns:
            z_auth: [B, N, K, V, D_auth]
            z_src:  [B, N, K, V, D_src]
        """
        B, N, K, V, _, P, _ = tubes_rgb.shape

        # Flatten all patches
        rgb_flat = tubes_rgb.reshape(B * N * K * V, 3, P, P)

        # GPU preprocessing
        sp_flat, fr_flat = self.preprocess_rgb(rgb_flat)

        # Single forward pass
        h_sp = self.enc_spatial(sp_flat)
        h_fr = self.enc_wavelet(fr_flat)

        h      = self.fusion(h_sp, h_fr)
        z_auth = self.proj_auth(h)
        z_src  = self.proj_src(h)

        # Reshape back
        z_auth = z_auth.view(B, N, K, V, self.z_auth_dim)
        z_src  = z_src.view(B, N, K, V, self.z_src_dim)

        return {"z_auth": z_auth, "z_src": z_src}

    def forward(
        self,
        tubes: torch.Tensor,
        tubes_wavelet: torch.Tensor,
        tube_mask: torch.Tensor = None,  # [B, N] bool mask
        view_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Classification forward (uses a single view — typically the original, view 0).

        Args:
            tubes:         [B, N, K, V, 6, P, P]
            tubes_wavelet: [B, N, K, V, 3, P, P]
            tube_mask:     [B, N] bool (True = valid tube, False = padding)
            view_idx:      view to encode (0 = no augmentation)

        Returns:
            z_auth:       [B, N, D_auth]   per-tube auth embeddings
            z_src:        [B, N, D_src]    per-tube src  embeddings
            a_local:      [B, N]            per-tube local auth scores (raw logit)
            h_img:        [B, D_fused]      image-level representation
            attn_weights: [B, N]            MIL attention weights
            logits_auth:  [B, 2]            binary classification logits
            logits_src:   [B, M] or None    source classification logits
        """
        B, N = tubes.shape[:2]

        enc = self.encode_patches(tubes, tubes_wavelet, view_idx)

        # Scale aggregation: average over K scales → per-tube representations
        # Re-normalize z_auth/z_src after averaging (mean of unit vectors ≠ unit vector)
        h_tube      = enc["h"].mean(dim=2)                              # [B, N, D_fused]
        z_auth_tube = F.normalize(enc["z_auth"].mean(dim=2), dim=-1)   # [B, N, D_auth]
        z_src_tube  = F.normalize(enc["z_src"].mean(dim=2),  dim=-1)   # [B, N, D_src]

        # Local authenticity score per tube (raw logit, used for MIL / top-k)
        a_local = self.local_score(z_auth_tube).squeeze(-1)  # [B, N]

        # MIL attention pooling over N tubes → image representation
        # Use tube_mask if provided, otherwise all tubes are valid
        if tube_mask is None:
            tube_mask = torch.ones(B, N, dtype=torch.bool, device=tubes.device)
        h_img, attn_weights = self.mil_agg(h_tube, tube_mask)  # [B, D_fused], [B, N]

        # Classification heads
        logits_auth = self.binary_head(h_img)
        logits_src  = self.source_head(h_img) if self.source_head is not None else None

        return {
            "z_auth":       z_auth_tube,   # [B, N, D_auth]
            "z_src":        z_src_tube,    # [B, N, D_src]
            "a_local":      a_local,       # [B, N]
            "h_img":        h_img,         # [B, D_fused]
            "attn_weights": attn_weights,  # [B, N]
            "logits_auth":  logits_auth,   # [B, 2]
            "logits_src":   logits_src,    # [B, M] or None
        }

    def forward_from_rgb(
        self,
        tubes_rgb: torch.Tensor,
        tube_mask: torch.Tensor = None,
        view_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Classification forward from RGB input (FastMultiScaleTubeCollator format).

        Args:
            tubes_rgb: [B, N, K, V, 3, P, P] RGB in [0, 1]
            tube_mask: [B, N] bool mask
            view_idx:  view to encode

        Returns:
            Same as forward()
        """
        B, N = tubes_rgb.shape[:2]

        enc = self.encode_patches_from_rgb(tubes_rgb, view_idx)

        h_tube      = enc["h"].mean(dim=2)
        z_auth_tube = F.normalize(enc["z_auth"].mean(dim=2), dim=-1)
        z_src_tube  = F.normalize(enc["z_src"].mean(dim=2),  dim=-1)

        a_local = self.local_score(z_auth_tube).squeeze(-1)

        if tube_mask is None:
            tube_mask = torch.ones(B, N, dtype=torch.bool, device=tubes_rgb.device)
        h_img, attn_weights = self.mil_agg(h_tube, tube_mask)

        logits_auth = self.binary_head(h_img)
        logits_src  = self.source_head(h_img) if self.source_head is not None else None

        return {
            "z_auth":       z_auth_tube,
            "z_src":        z_src_tube,
            "a_local":      a_local,
            "h_img":        h_img,
            "attn_weights": attn_weights,
            "logits_auth":  logits_auth,
            "logits_src":   logits_src,
        }


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class TubeContrastiveModule(pl.LightningModule):
    """
    Lightning wrapper for TubeModel.

    Two training phases:
      - phase=1 : pure contrastive training (Phase1Loss — SupCon auth + SupCon src + decoupling)
      - phase=2 : classification fine-tuning (BCE auth + optional CE src)

    Supports two input formats:
      - Legacy: tubes [B,N,K,V,6,P,P] + tubes_wavelet [B,N,K,V,3,P,P] (CPU preprocessing)
      - Fast:   tubes_rgb [B,N,K,V,3,P,P] (GPU preprocessing via FastMultiScaleTubeCollator)

    Args:
        encoder_dim:        output dim of each PatchEncoder branch
        fused_dim:          dim after FusionMLP
        z_auth_dim:         auth projection space dim
        z_src_dim:          src  projection space dim
        attn_dim:           dim for MIL attention MLP
        num_src_classes:    number of generator classes (None → no source head)
        pretrained_spatial: use ImageNet weights for the 6-ch spatial encoder
        lr:                 peak learning rate
        warmup_steps:       linear warm-up steps
        predict_model:      whether to predict the source/model label
        phase:              training phase (1=contrastive, 2=classification)
        target_size:        patch size (for GPU preprocessing)
        lambda_auth:        [phase 1] weight for auth SupCon loss
        lambda_src_con:     [phase 1] weight for src  SupCon loss
        lambda_decouple:    [phase 1] weight for decoupling penalty
        temp_auth:          [phase 1] temperature τ for auth SupCon
        temp_src:           [phase 1] temperature τ for src  SupCon
        lambda_src_cls:     [phase 2] weight for source classification loss
    """

    def __init__(
        self,
        encoder_dim: int = 256,
        fused_dim: int = 256,
        z_auth_dim: int = 128,
        z_src_dim: int = 128,
        attn_dim: int = 128,
        num_src_classes: Optional[int] = None,
        pretrained_spatial: bool = False,
        lr: float = 5e-4,
        warmup_steps: int = 200,
        predict_model: bool = False,
        # Phase selection
        phase: int = 1,
        # Target size for GPU preprocessing
        target_size: int = 96,
        # Phase 1 loss hyperparameters
        lambda_auth: float = 1.0,
        lambda_src_con: float = 0.5,
        lambda_decouple: float = 0.01,
        temp_auth: float = 0.07,
        temp_src: float = 0.07,
        # Phase 2 loss hyperparameters
        lambda_src_cls: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.phase = phase
        self.predict_model = predict_model
        self.lambda_src_cls = lambda_src_cls

        self.model = TubeModel(
            encoder_dim=encoder_dim,
            fused_dim=fused_dim,
            z_auth_dim=z_auth_dim,
            z_src_dim=z_src_dim,
            attn_dim=attn_dim,
            num_src_classes=num_src_classes,
            pretrained_spatial=pretrained_spatial,
            target_size=target_size,
        )

        # Phase 1: contrastive losses
        self.phase1_loss = Phase1Loss(
            lambda_auth=lambda_auth,
            lambda_src=lambda_src_con,
            lambda_decouple=lambda_decouple,
            temp_auth=temp_auth,
            temp_src=temp_src,
        )

        # Phase 2: classification losses
        self.loss_auth = nn.CrossEntropyLoss()
        self.loss_src  = nn.CrossEntropyLoss() if predict_model else None

        self.val_outputs: list = []

    # -------------------------------------------------------------------------

    def freeze_layers(
        self,
        freeze_encoders: bool = True,
        freeze_fusion: bool = False,
        freeze_projections: bool = True,
    ):
        """
        Freeze specific layers for phase 2 fine-tuning.

        Args:
            freeze_encoders:    freeze enc_spatial and enc_wavelet (recommended)
            freeze_fusion:      freeze FusionMLP (False → train it in phase 2)
            freeze_projections: freeze proj_auth and proj_src (True → not used in phase 2)

        Always trainable in phase 2:
            - mil_agg (AttnAggregator)
            - binary_head
            - source_head (if predict_model=True)
        """
        if freeze_encoders:
            for param in self.model.enc_spatial.parameters():
                param.requires_grad = False
            for param in self.model.enc_wavelet.parameters():
                param.requires_grad = False

        if freeze_fusion:
            for param in self.model.fusion.parameters():
                param.requires_grad = False

        if freeze_projections:
            for param in self.model.proj_auth.parameters():
                param.requires_grad = False
            for param in self.model.proj_src.parameters():
                param.requires_grad = False
            for param in self.model.local_score.parameters():
                param.requires_grad = False

        # Also freeze GPU preprocessor (no trainable params, but good practice)
        for param in self.model.gpu_preproc.parameters():
            param.requires_grad = False

        # Log trainable params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[freeze_layers] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(self, tubes, tubes_wavelet=None, tube_mask=None, view_idx: int = 0):
        """Forward with auto-detection of input format."""
        if tubes_wavelet is not None:
            # Legacy format: tubes [B,N,K,V,6,P,P] + tubes_wavelet [B,N,K,V,3,P,P]
            return self.model(tubes, tubes_wavelet, tube_mask=tube_mask, view_idx=view_idx)
        else:
            # Fast format: tubes is actually tubes_rgb [B,N,K,V,3,P,P]
            return self.model.forward_from_rgb(tubes, tube_mask=tube_mask, view_idx=view_idx)

    def _get_input_tensors(self, batch: Dict[str, Any]):
        """Extract input tensors from batch, handling both formats."""
        if "tubes_rgb" in batch:
            # Fast format (FastMultiScaleTubeCollator)
            return batch["tubes_rgb"], None, True
        else:
            # Legacy format (MultiScaleTubeCollator)
            return batch["tubes"], batch["tubes_wavelet"], False

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        tubes, wav, is_rgb_format = self._get_input_tensors(batch)
        labels = batch["labels"]
        tube_mask = batch.get("tube_mask", None)
        model_labels = batch.get("model_labels", None)

        if self.phase == 1:
            # ── Pure contrastive phase ─────────────────────────────────────────
            B, N, K, V = tubes.shape[:4]

            if is_rgb_format:
                enc = self.model.encode_per_scale_all_views_from_rgb(tubes)
            else:
                enc = self.model.encode_per_scale_all_views(tubes, wav)

            losses = self.phase1_loss(
                enc["z_auth"], enc["z_src"],
                labels, model_labels,
                B, N, K, V,
                tube_mask=tube_mask,
            )
            loss = losses["loss"]
            self.log("train/loss",          loss,                    prog_bar=True,  on_step=True, on_epoch=True)
            self.log("train/loss_auth",     losses["loss_auth"],     prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/loss_src_con",  losses["loss_src"],      prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/loss_decouple", losses["loss_decouple"], prog_bar=False, on_step=True, on_epoch=True)

        else:
            # ── Classification phase ───────────────────────────────────────────
            out = self(tubes, wav, tube_mask=tube_mask, view_idx=0)
            loss = self.loss_auth(out["logits_auth"], labels)
            self.log("train/loss",      loss, prog_bar=True,  on_step=True, on_epoch=True)
            self.log("train/loss_auth", loss, prog_bar=False, on_step=True, on_epoch=True)

            if (
                self.predict_model
                and "model_labels" in batch
                and out["logits_src"] is not None
            ):
                syn_mask = labels == 1
                if syn_mask.sum() > 0:
                    loss_src_cls = self.loss_src(
                        out["logits_src"][syn_mask],
                        batch["model_labels"][syn_mask],
                    )
                    loss = loss + self.lambda_src_cls * loss_src_cls
                    self.log("train/loss_src_cls", loss_src_cls, prog_bar=False, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        tubes, wav, is_rgb_format = self._get_input_tensors(batch)
        labels = batch["labels"]
        tube_mask = batch.get("tube_mask", None)
        model_labels = batch.get("model_labels", None)

        if self.phase == 1:
            # ── Contrastive val loss ───────────────────────────────────────────
            B, N, K, V = tubes.shape[:4]

            if is_rgb_format:
                enc = self.model.encode_per_scale_all_views_from_rgb(tubes)
            else:
                enc = self.model.encode_per_scale_all_views(tubes, wav)

            losses = self.phase1_loss(
                enc["z_auth"], enc["z_src"],
                labels, model_labels,
                B, N, K, V,
                tube_mask=tube_mask,
            )
            loss = losses["loss"]
            self.log("val/loss",          loss,                    prog_bar=True,  on_epoch=True, on_step=False)
            self.log("val/loss_auth",     losses["loss_auth"],     prog_bar=False, on_epoch=True, on_step=False)
            self.log("val/loss_src_con",  losses["loss_src"],      prog_bar=False, on_epoch=True, on_step=False)
            self.log("val/loss_decouple", losses["loss_decouple"], prog_bar=False, on_epoch=True, on_step=False)
            self.val_outputs.append({"loss": loss.detach()})

        else:
            # ── Classification val ─────────────────────────────────────────────
            out = self(tubes, wav, tube_mask=tube_mask, view_idx=0)
            loss = self.loss_auth(out["logits_auth"], labels)

            preds = out["logits_auth"].argmax(dim=-1)
            acc   = (preds == labels).float().mean()

            self.log("val/loss", loss, prog_bar=True,  on_epoch=True, on_step=False)
            self.log("val/acc",  acc,  prog_bar=True,  on_epoch=True, on_step=False)

            probs = torch.softmax(out["logits_auth"], dim=-1)[:, 1]
            self.val_outputs.append({
                "loss":   loss.detach(),
                "labels": labels.detach(),
                "probs":  probs.detach(),
            })

        return loss

    def on_validation_epoch_end(self):
        outputs = self.val_outputs
        self.val_outputs = []

        if not outputs or self.phase == 1:
            return

        all_labels = np.concatenate([o["labels"].cpu().numpy() for o in outputs])
        all_probs  = np.concatenate([o["probs"].cpu().numpy()  for o in outputs])

        auc = compute_binary_auc(all_labels, all_probs)
        if not np.isnan(auc):
            self.log("val/auc", auc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        no_decay_keywords = {"bias", "LayerNorm", "layer_norm", "bn."}
        params_decay    = []
        params_no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in no_decay_keywords):
                params_no_decay.append(param)
            else:
                params_decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": params_decay,    "weight_decay": 1e-2},
                {"params": params_no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        warmup_steps = self.hparams.warmup_steps
        total_steps  = self.trainer.estimated_stepping_batches

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
