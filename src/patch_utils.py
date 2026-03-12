"""Utilities for patch extraction from full images."""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F


def sliding_window_indices(H: int, W: int, patch: int, stride: int) -> List[Tuple[int, int]]:
    """
    Generate top-left indices for a sliding window over an HxW image.
    We ensure at least one index (0, 0) if the image is smaller.
    
    Args:
        H: Image height
        W: Image width
        patch: Patch size
        stride: Stride between patches
        
    Returns:
        List of (y, x) top-left coordinates
    """
    ys = list(range(0, max(H - patch, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - patch, 0) + 1, stride)) or [0]
    return [(y, x) for y in ys for x in xs]


def extract_patches_tensor_auto(
    x: torch.Tensor,
    patch: int,
    max_patches: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract patches with an automatically chosen stride so that
    we get roughly <= max_patches. If we still get more, randomly
    subsample.

    Args:
        x:          Tensor [C,H,W]
        patch:      Patch size
        max_patches:Max number of patches (approximate)

    Returns:
        patches: [N,C,patch,patch]
        coords:  [N,2] with normalized centers (cy, cx) in [0,1]
    """
    C, H, W = x.shape

    # Choose stride based on a target grid size ~ sqrt(max_patches)
    grid = max(1, int(math.sqrt(max_patches)))
    stride = max(1, min(H, W) // grid)

    coords, patches = [], []

    # Sliding window over the full image
    for (y, x0) in sliding_window_indices(H, W, patch, stride):
        y2, x2 = y + patch, x0 + patch
        yy2, xx2 = min(y2, H), min(x2, W)

        crop = x[:, y:yy2, x0:xx2]

        # Pad if near border (right / bottom)
        pad_h = patch - crop.shape[1]
        pad_w = patch - crop.shape[2]
        if pad_h > 0 or pad_w > 0:
            # pad format: (left, right, top, bottom)
            crop = F.pad(crop, (0, pad_w, 0, pad_h))

        patches.append(crop)

        # Center (in pixels)
        cy_pix = min(y + patch / 2, H)
        cx_pix = min(x0 + patch / 2, W)

        # Normalize by original H,W
        cy = cy_pix / max(H, 1)
        cx = cx_pix / max(W, 1)
        coords.append([cy, cx])

    # Fallback if something weird happens (should not occur with current logic)
    if not patches:
        patches = [x]
        coords = [[0.5, 0.5]]

    patches = torch.stack(patches, dim=0)
    coords = torch.tensor(coords, dtype=torch.float32)

    # If we got too many patches, randomly subsample
    N = patches.shape[0]
    if N > max_patches:
        idx = torch.randperm(N)[:max_patches]
        patches = patches[idx]
        coords = coords[idx]

    return patches, coords
