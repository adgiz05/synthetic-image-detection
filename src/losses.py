"""
Contrastive losses for multi-scale tube forensics — Phase 1.

Three losses are combined into a single Phase1Loss module:

  L = λ_auth · L_supcon_auth  +  λ_src · L_supcon_src  +  λ_decouple · L_decouple

  1. L_supcon_auth
     ─────────────
     Hierarchical Weighted SupCon on z_auth [B·N·K·V, D].
     Positive weight hierarchy (operates on per-scale, per-view embeddings):
       • w_view  (default 1.0) : same spatial center, same scale, different view
       • w_scale (default 0.8) : same spatial center, different scale  (any view)
       • w_tube  (default 0.3) : same image, different spatial center

  2. L_supcon_src
     ─────────────
     Standard SupCon on z_src, restricted to synthetic embeddings.
     Positives: same generator ID.
     Skipped entirely when there are no synthetic samples in the batch.

  3. L_decouple
     ───────────
     Minimise linear correlation between z_auth and z_src spaces:
       L_decouple = ‖ Corr(z_auth, z_src) ‖²_F
     where Corr is the empirical cross-correlation matrix computed over the batch.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _build_index_tensors(
    B: int, N: int, K: int, V: int, device: torch.device
):
    """Return flat (M,) index tensors for each dimension, where M = B·N·K·V."""
    M = B * N * K * V
    b = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, K, V).reshape(M)
    n = torch.arange(N, device=device).view(1, N, 1, 1).expand(B, N, K, V).reshape(M)
    k = torch.arange(K, device=device).view(1, 1, K, 1).expand(B, N, K, V).reshape(M)
    v = torch.arange(V, device=device).view(1, 1, 1, V).expand(B, N, K, V).reshape(M)
    return b, n, k, v


def build_auth_weight_matrix(
    B: int,
    N: int,
    K: int,
    V: int,
    w_view: float = 1.0,
    w_scale: float = 0.8,
    w_tube: float = 0.3,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build the [M, M] positive-weight matrix for the auth SupCon,
    where M = B·N·K·V.

    Relationship                  condition                              weight
    ─────────────────────────────────────────────────────────────────────────────
    same patch, diff view       same (b, n, k), different v            w_view
    same center, diff scale     same (b, n), different k               w_scale
    same image, diff center     same b, different n                    w_tube
    different image / self      different b  or  i==j                  0

    Weights are additive — in practice each pair triggers at most one rule.
    """
    M = B * N * K * V
    b, n, k, v = _build_index_tensors(B, N, K, V, device)

    same_b = (b[:, None] == b[None, :])  # [M, M]
    same_n = (n[:, None] == n[None, :])
    same_k = (k[:, None] == k[None, :])
    same_v = (v[:, None] == v[None, :])
    is_self = torch.eye(M, dtype=torch.bool, device=device)

    W = torch.zeros(M, M, device=device)
    W += w_view  * (same_b & same_n & same_k & ~same_v & ~is_self).float()
    W += w_scale * (same_b & same_n & ~same_k              & ~is_self).float()
    W += w_tube  * (same_b & ~same_n                       & ~is_self).float()
    return W


# ---------------------------------------------------------------------------
# Generic weighted SupCon
# ---------------------------------------------------------------------------

def weighted_supcon(
    z: torch.Tensor,
    pos_weights: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Weighted Supervised Contrastive Loss.

    For each anchor i:
        L_i = −( Σ_p w_ip · (z_i·z_p / τ  −  log Σ_{a≠i} exp(z_i·z_a / τ)) )
              / Σ_p w_ip

    Averaged over anchors that have at least one positive.

    Args:
        z:           [M, D]  L2-normalized embeddings
        pos_weights: [M, M]  ≥ 0; entry (i, j) = positive weight for pair (i, j)
        temperature: τ

    Returns:
        Scalar loss.
    """
    M = z.size(0)
    device = z.device

    sim = torch.mm(z, z.T) / temperature           # [M, M]

    # log Σ_{a≠i} exp(z_i · z_a / τ)
    mask_self = torch.eye(M, dtype=torch.bool, device=device)
    log_denom = torch.logsumexp(
        sim.masked_fill(mask_self, float("-inf")), dim=1
    )                                               # [M]

    # log p(positive | anchor i): sim[i,j] - log_denom[i]
    log_probs = sim - log_denom.unsqueeze(1)        # [M, M]

    # Weighted average over positives
    weight_sum = pos_weights.sum(dim=1).clamp(min=1e-8)  # [M]
    loss_per_anchor = -(pos_weights * log_probs).sum(dim=1) / weight_sum  # [M]

    has_pos = pos_weights.sum(dim=1) > 0
    if not has_pos.any():
        return z.new_zeros(1).squeeze()

    return loss_per_anchor[has_pos].mean()


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------

def supcon_auth_loss(
    z_auth: torch.Tensor,
    B: int,
    N: int,
    K: int,
    V: int,
    w_view: float = 1.0,
    w_scale: float = 0.8,
    w_tube: float = 0.3,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Hierarchical SupCon for authenticity embeddings.

    Args:
        z_auth:      [M, D_auth] L2-normalized, M = B·N·K·V
        B, N, K, V:  batch dimensions
        w_*:         positive weight per relationship level
        temperature: SupCon τ

    Returns:
        Scalar loss.
    """
    W = build_auth_weight_matrix(B, N, K, V, w_view, w_scale, w_tube, z_auth.device)
    return weighted_supcon(z_auth, W, temperature)


def supcon_src_loss(
    z_src: torch.Tensor,
    auth_labels: torch.Tensor,
    model_labels: torch.Tensor,
    B: int,
    N: int,
    K: int,
    V: int,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    SupCon for source/generator embeddings, restricted to synthetic patches.

    Positives  : same generator (model_label).
    Denominator: all synthetic patches in the batch (including same generator).

    Args:
        z_src:        [M, D_src] L2-normalized, M = B·N·K·V
        auth_labels:  [B]  0=real, 1=synthetic
        model_labels: [B]  generator ID (only meaningful for synthetic images)
        B, N, K, V:   batch dimensions
        temperature:  SupCon τ

    Returns:
        Scalar loss (0 if no synthetic samples in batch).
    """
    M = z_src.size(0)
    device = z_src.device

    # Expand image-level labels to per-embedding level
    b_idx, _, _, _ = _build_index_tensors(B, N, K, V, device)
    auth_emb  = auth_labels[b_idx]    # [M]
    model_emb = model_labels[b_idx]   # [M]

    syn_mask = auth_emb == 1
    if not syn_mask.any():
        return z_src.new_zeros(1).squeeze()

    z_syn   = z_src[syn_mask]          # [M_syn, D]
    gen_syn = model_emb[syn_mask]      # [M_syn]
    M_syn = z_syn.size(0)

    same_gen = (gen_syn[:, None] == gen_syn[None, :])         # [M_syn, M_syn]
    is_self  = torch.eye(M_syn, dtype=torch.bool, device=device)
    W = (same_gen & ~is_self).float()

    return weighted_supcon(z_syn, W, temperature)


def decoupling_loss(
    z_auth: torch.Tensor,
    z_src: torch.Tensor,
) -> torch.Tensor:
    """
    Penalise linear correlation between z_auth and z_src:

        L_decouple = ‖ C ‖²_F,   C_ij = corr(z_auth_col_i, z_src_col_j)

    Args:
        z_auth: [M, D_auth]
        z_src:  [M, D_src]

    Returns:
        Scalar loss.
    """
    M = z_auth.size(0)

    # Column-wise standardisation (zero mean, unit variance)
    z_a = (z_auth - z_auth.mean(0)) / (z_auth.std(0) + 1e-8)  # [M, D_auth]
    z_s = (z_src  - z_src.mean(0))  / (z_src.std(0)  + 1e-8)  # [M, D_src]

    C = (z_a.T @ z_s) / M   # [D_auth, D_src]  empirical cross-correlation
    return (C ** 2).sum()


# ---------------------------------------------------------------------------
# Combined Phase 1 loss
# ---------------------------------------------------------------------------

class Phase1Loss(nn.Module):
    """
    Phase 1 (pure contrastive) training loss:

        L = λ_auth · L_supcon_auth  +  λ_src · L_supcon_src  +  λ_decouple · L_decouple

    Args:
        lambda_auth:     weight for auth SupCon
        lambda_src:      weight for src  SupCon
        lambda_decouple: weight for decoupling penalty
        w_view:          same center / same scale / different view  (positive weight)
        w_scale:         same center / different scale              (positive weight)
        w_tube:          same image  / different center             (positive weight)
        temp_auth:       temperature τ for auth SupCon
        temp_src:        temperature τ for src  SupCon
    """

    def __init__(
        self,
        lambda_auth: float = 1.0,
        lambda_src: float = 0.5,
        lambda_decouple: float = 0.1,
        w_view: float = 1.0,
        w_scale: float = 0.8,
        w_tube: float = 0.3,
        temp_auth: float = 0.07,
        temp_src: float = 0.07,
    ):
        super().__init__()
        self.lambda_auth     = lambda_auth
        self.lambda_src      = lambda_src
        self.lambda_decouple = lambda_decouple
        self.w_view          = w_view
        self.w_scale         = w_scale
        self.w_tube          = w_tube
        self.temp_auth       = temp_auth
        self.temp_src        = temp_src

    def forward(
        self,
        z_auth: torch.Tensor,
        z_src: torch.Tensor,
        auth_labels: torch.Tensor,
        model_labels: Optional[torch.Tensor],
        B: int,
        N: int,
        K: int,
        V: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_auth:       [B, N, K, V, D_auth]  — per-scale per-view auth embeddings
            z_src:        [B, N, K, V, D_src]   — per-scale per-view src  embeddings
            auth_labels:  [B]  0=real, 1=synthetic
            model_labels: [B]  generator ID, or None (src loss skipped if None)
            B, N, K, V:   batch dimensions (must match z_auth.shape[:4])

        Returns:
            Dict:
                loss           — total weighted scalar loss
                loss_auth      — L_supcon_auth
                loss_src       — L_supcon_src  (0 if no synthetic / no model_labels)
                loss_decouple  — L_decouple
        """
        M = B * N * K * V

        # Flatten and re-normalise (mean of unit vectors ≠ unit vector)
        z_a = F.normalize(z_auth.reshape(M, -1), dim=-1)  # [M, D_auth]
        z_s = F.normalize(z_src.reshape(M, -1),  dim=-1)  # [M, D_src]

        # ── Auth SupCon ──────────────────────────────────────────────────────
        l_auth = supcon_auth_loss(
            z_a, B, N, K, V,
            self.w_view, self.w_scale, self.w_tube,
            self.temp_auth,
        )

        # ── Src SupCon (skip if model_labels not available) ──────────────────
        if model_labels is not None:
            l_src = supcon_src_loss(
                z_s, auth_labels, model_labels,
                B, N, K, V, self.temp_src,
            )
        else:
            l_src = z_a.new_zeros(1).squeeze()

        # ── Decoupling ───────────────────────────────────────────────────────
        l_decouple = decoupling_loss(z_a, z_s)

        total = (
            self.lambda_auth     * l_auth
            + self.lambda_src    * l_src
            + self.lambda_decouple * l_decouple
        )

        return {
            "loss":          total,
            "loss_auth":     l_auth,
            "loss_src":      l_src,
            "loss_decouple": l_decouple,
        }
