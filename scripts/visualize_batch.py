"""
Script to visualize multi-scale tube batches.

Plots per batch:
  1. Original images with tube extraction regions overlaid
  2. Per-tube full representation: RGB | Residual | Wavelet(LH/HL/HH)  for every (scale, view)
  3. Scale comparison (RGB only)
  4. View comparison (RGB only)

Usage:
    python scripts/visualize_batch.py --train_csv data/train.csv --root_dir data/
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets import MultiScaleTubeDataset
from src.collators import MultiScaleTubeCollator
from src.constants import IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverse ImageNet normalization on a [3, H, W] or [B, 3, H, W] tensor."""
    tensor = tensor.clone()
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_t  = torch.tensor(std,  dtype=tensor.dtype).view(-1, 1, 1)
    if tensor.dim() == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t  = std_t.unsqueeze(0)
    return tensor * std_t + mean_t


def rgb_to_image(tensor_3ch, denorm=True):
    """
    [3, H, W] float tensor → [H, W, 3] numpy in [0, 1].
    Expects the first 3 channels to be ImageNet-normalized RGB.
    """
    t = tensor_3ch.detach().cpu()
    if denorm:
        t = denormalize(t)
    img = t.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def residual_to_image(tensor_3ch, amplify=3.0):
    """
    [3, H, W] zero-centered residual → [H, W, 3] numpy in [0, 1].
    Shifts to gray=0.5 and amplifies so subtle artifacts become visible.
    """
    t = tensor_3ch.detach().cpu()
    img = (t * amplify + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def wavelet_to_image(tensor_3ch):
    """
    [3, H, W] wavelet bands (LH/HL/HH, range ≈ [-3, 3]) → [H, W, 3] false-color numpy.
    R=LH (horizontal edges), G=HL (vertical edges), B=HH (diagonal).
    """
    t = tensor_3ch.detach().cpu()
    img = ((t + 3.0) / 6.0).clamp(0, 1).permute(1, 2, 0).numpy()
    return img


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_tube_representations(
    tube_spatial,
    tube_wavelet,
    tube_idx,
    image_idx,
    label,
    center,
    scales,
    output_dir,
    denorm=True,
):
    """
    Comprehensive view of one tube across all scales and views.

    Grid layout: rows = K scales, cols = V views × 3 (RGB | Residual | Wavelet)

    Args:
        tube_spatial: [K, V, 6, P, P]  – first 3ch RGB (norm), last 3ch residual
        tube_wavelet: [K, V, 3, P, P]  – LH / HL / HH bands
        scales: list of scale values (e.g. [64, 128, 256])
    """
    K, V, _, P, _ = tube_spatial.shape

    ncols = V * 3   # per view: RGB, Residual, Wavelet
    nrows = K

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.8))

    # Ensure 2-D axes array
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        f"Image {image_idx}  |  Tube {tube_idx}  |  "
        f"Label: {'Synthetic' if label == 1 else 'Real'}  |  "
        f"Center: ({center[0]:.2f}, {center[1]:.2f})\n"
        f"Columns per view:  RGB  |  Residual (×3 ampl.)  |  Wavelet LH/HL/HH (false-color)",
        fontsize=11, fontweight='bold',
    )

    for k in range(K):
        scale_label = f"{scales[k]}px" if k < len(scales) else f"s{k}"
        for v in range(V):
            col_base = v * 3

            patch_sp = tube_spatial[k, v]   # [6, P, P]
            rgb = patch_sp[:3]              # [3, P, P]
            res = patch_sp[3:]              # [3, P, P]
            wav = tube_wavelet[k, v]        # [3, P, P]

            view_str = "orig" if v == 0 else f"aug{v}"

            # RGB
            ax = axes[k, col_base]
            ax.imshow(rgb_to_image(rgb, denorm=denorm))
            ax.axis('off')
            ax.set_title(f"s{k}({scale_label}) v{v}\nRGB [{view_str}]", fontsize=8)

            # Residual
            ax = axes[k, col_base + 1]
            ax.imshow(residual_to_image(res))
            ax.axis('off')
            ax.set_title(f"s{k}({scale_label}) v{v}\nResidual", fontsize=8)

            # Wavelet
            ax = axes[k, col_base + 2]
            ax.imshow(wavelet_to_image(wav))
            ax.axis('off')
            ax.set_title(f"s{k}({scale_label}) v{v}\nWavelet", fontsize=8)

    plt.tight_layout()

    image_dir = Path(output_dir) / f"image_{image_idx}"
    image_dir.mkdir(parents=True, exist_ok=True)
    save_path = image_dir / f"tube_{tube_idx}_representations.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_wavelet_bands(
    tube_wavelet,
    tube_idx,
    image_idx,
    label,
    center,
    scales,
    output_dir,
):
    """
    Show each wavelet band (LH, HL, HH) separately as grayscale for every
    (scale, view), using a diverging colormap to make edges pop.

    Grid: rows = K scales, cols = V * 3 bands
    """
    K, V, _, P, _ = tube_wavelet.shape
    band_names = ["LH (horiz)", "HL (vert)", "HH (diag)"]

    ncols = V * 3
    nrows = K

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.8))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        f"Wavelet bands  |  Image {image_idx}  |  Tube {tube_idx}  |  "
        f"Label: {'Synthetic' if label == 1 else 'Real'}",
        fontsize=11, fontweight='bold',
    )

    for k in range(K):
        scale_label = f"{scales[k]}px" if k < len(scales) else f"s{k}"
        for v in range(V):
            for b in range(3):
                col = v * 3 + b
                band = tube_wavelet[k, v, b].detach().cpu().numpy()  # [P, P]
                ax = axes[k, col]
                ax.imshow(band, cmap='RdBu_r', vmin=-3, vmax=3)
                ax.axis('off')
                ax.set_title(
                    f"s{k}({scale_label}) v{v}\n{band_names[b]}", fontsize=7
                )

    plt.tight_layout()

    image_dir = Path(output_dir) / f"image_{image_idx}"
    image_dir.mkdir(parents=True, exist_ok=True)
    save_path = image_dir / f"tube_{tube_idx}_wavelet_bands.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_batch_summary(
    batch,
    num_samples=2,
    num_tubes_per_sample=4,
    output_dir="visualizations",
    denorm=True,
):
    """
    For each selected image, plot the full representation of a subset of tubes.
    """
    tubes    = batch['tubes']           # [B, N, K, V, 6, P, P]
    wav      = batch['tubes_wavelet']   # [B, N, K, V, 3, P, P]
    centers  = batch['tube_centers']    # [B, N, 2]
    labels   = batch['labels']          # [B]
    scales   = batch.get('scales', [])

    B, N, K, V, C, P, _ = tubes.shape

    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"Batch size:         {B}")
    print(f"Tubes per image:    {N}")
    print(f"Scales per tube:    {K}  {scales}")
    print(f"Views per scale:    {V}")
    print(f"Patch size:         {P}×{P}")
    print(f"Spatial channels:   {C}  (3 RGB + 3 residual)")
    print(f"Wavelet channels:   {wav.shape[4]}  (LH / HL / HH)")
    print(f"Labels:             {labels.tolist()}")
    if 'model_labels' in batch:
        print(f"Model labels:       {batch['model_labels'].tolist()}")
    print(f"{'='*60}\n")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    num_samples = min(num_samples, B)
    for b in range(num_samples):
        label = labels[b].item()
        tube_indices = np.linspace(0, N - 1, min(num_tubes_per_sample, N), dtype=int)

        for tube_idx in tube_indices:
            center = centers[b, tube_idx].tolist()

            # Full representation: RGB + residual + wavelet
            plot_tube_representations(
                tube_spatial=tubes[b, tube_idx],
                tube_wavelet=wav[b, tube_idx],
                tube_idx=tube_idx,
                image_idx=b,
                label=label,
                center=center,
                scales=scales,
                output_dir=output_dir,
                denorm=denorm,
            )

            # Individual wavelet bands with diverging colormap
            plot_wavelet_bands(
                tube_wavelet=wav[b, tube_idx],
                tube_idx=tube_idx,
                image_idx=b,
                label=label,
                center=center,
                scales=scales,
                output_dir=output_dir,
            )


def plot_scale_comparison(
    batch,
    image_idx=0,
    tube_idx=0,
    view_idx=0,
    output_dir="visualizations",
    denorm=True,
):
    """
    For one (tube, view), compare all scales side by side: RGB | Residual | Wavelet.
    """
    tubes   = batch['tubes']           # [B, N, K, V, 6, P, P]
    wav     = batch['tubes_wavelet']   # [B, N, K, V, 3, P, P]
    centers = batch['tube_centers']
    labels  = batch['labels']
    scales  = batch.get('scales', [])

    K = tubes.shape[2]

    tube_sp  = tubes[image_idx, tube_idx]   # [K, V, 6, P, P]
    tube_wav = wav[image_idx, tube_idx]     # [K, V, 3, P, P]
    center   = centers[image_idx, tube_idx].tolist()
    label    = labels[image_idx].item()

    ncols = K * 3   # per scale: RGB, Residual, Wavelet
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.8, 3.5))
    if ncols == 1:
        axes = [axes]

    fig.suptitle(
        f"Scale comparison  |  Image {image_idx}  |  Tube {tube_idx}  |  View {view_idx}\n"
        f"Label: {'Synthetic' if label == 1 else 'Real'}  |  "
        f"Center: ({center[0]:.2f}, {center[1]:.2f})",
        fontsize=12, fontweight='bold',
    )

    for k in range(K):
        scale_label = f"{scales[k]}px" if k < len(scales) else f"s{k}"
        patch_sp  = tube_sp[k, view_idx]   # [6, P, P]
        patch_wav = tube_wav[k, view_idx]  # [3, P, P]

        axes[k * 3 + 0].imshow(rgb_to_image(patch_sp[:3], denorm=denorm))
        axes[k * 3 + 0].set_title(f"s{k} ({scale_label})\nRGB", fontsize=9)
        axes[k * 3 + 0].axis('off')

        axes[k * 3 + 1].imshow(residual_to_image(patch_sp[3:]))
        axes[k * 3 + 1].set_title(f"s{k} ({scale_label})\nResidual", fontsize=9)
        axes[k * 3 + 1].axis('off')

        axes[k * 3 + 2].imshow(wavelet_to_image(patch_wav))
        axes[k * 3 + 2].set_title(f"s{k} ({scale_label})\nWavelet", fontsize=9)
        axes[k * 3 + 2].axis('off')

    plt.tight_layout()

    image_dir = Path(output_dir) / f"image_{image_idx}"
    image_dir.mkdir(parents=True, exist_ok=True)
    save_path = image_dir / f"scale_comparison_tube{tube_idx}_view{view_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_view_comparison(
    batch,
    image_idx=0,
    tube_idx=0,
    scale_idx=1,
    output_dir="visualizations",
    denorm=True,
):
    """
    For one (tube, scale), compare all views: RGB | Residual | Wavelet.
    """
    tubes   = batch['tubes']
    wav     = batch['tubes_wavelet']
    centers = batch['tube_centers']
    labels  = batch['labels']

    V = tubes.shape[3]

    tube_sp  = tubes[image_idx, tube_idx]
    tube_wav = wav[image_idx, tube_idx]
    center   = centers[image_idx, tube_idx].tolist()
    label    = labels[image_idx].item()

    ncols = V * 3
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.8, 3.5))
    if ncols == 1:
        axes = [axes]

    fig.suptitle(
        f"View comparison  |  Image {image_idx}  |  Tube {tube_idx}  |  Scale {scale_idx}\n"
        f"Label: {'Synthetic' if label == 1 else 'Real'}  |  "
        f"Center: ({center[0]:.2f}, {center[1]:.2f})",
        fontsize=12, fontweight='bold',
    )

    for v in range(V):
        patch_sp  = tube_sp[scale_idx, v]
        patch_wav = tube_wav[scale_idx, v]
        view_str  = "original" if v == 0 else f"augmented {v}"

        axes[v * 3 + 0].imshow(rgb_to_image(patch_sp[:3], denorm=denorm))
        axes[v * 3 + 0].set_title(f"v{v} ({view_str})\nRGB", fontsize=9)
        axes[v * 3 + 0].axis('off')

        axes[v * 3 + 1].imshow(residual_to_image(patch_sp[3:]))
        axes[v * 3 + 1].set_title(f"v{v} ({view_str})\nResidual", fontsize=9)
        axes[v * 3 + 1].axis('off')

        axes[v * 3 + 2].imshow(wavelet_to_image(patch_wav))
        axes[v * 3 + 2].set_title(f"v{v} ({view_str})\nWavelet", fontsize=9)
        axes[v * 3 + 2].axis('off')

    plt.tight_layout()

    image_dir = Path(output_dir) / f"image_{image_idx}"
    image_dir.mkdir(parents=True, exist_ok=True)
    save_path = image_dir / f"view_comparison_tube{tube_idx}_scale{scale_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_original_images_with_tube_centers(
    batch,
    num_samples=2,
    output_dir="visualizations",
):
    """
    Plot original images with tube extraction regions overlaid.
    """
    image_paths = batch.get('image_paths', [])
    centers = batch['tube_centers']
    labels  = batch['labels']
    scales  = batch.get('scales', [64, 128, 256])

    if not image_paths:
        print("Warning: batch does not contain 'image_paths', skipping original image plot")
        return

    B, N, _ = centers.shape
    num_samples = min(num_samples, B, len(image_paths))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nPlotting original images with tube centers...")

    for b in range(num_samples):
        img_path = image_paths[b]
        label    = labels[b].item()

        try:
            img       = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            H, W      = img_array.shape[:2]
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(img_array)

        colors = ['red', 'orange', 'yellow']
        for tube_idx in range(N):
            cy_norm, cx_norm = centers[b, tube_idx].tolist()
            cy = cy_norm * H
            cx = cx_norm * W

            ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2)

            for si, scale in enumerate(scales[:3]):
                half = scale / 2
                rect = mpatches.Rectangle(
                    (cx - half, cy - half), scale, scale,
                    color=colors[si % len(colors)],
                    fill=False, linewidth=1.5, alpha=0.6, linestyle='--',
                )
                ax.add_patch(rect)

            ax.text(
                cx + 10, cy - 10, f'T{tube_idx}',
                color='white', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
            )

        ax.set_title(
            f"Image {b}  |  Label: {'Synthetic' if label == 1 else 'Real'}\n"
            f"{N} tube extraction locations",
            fontsize=14, fontweight='bold',
        )
        ax.axis('off')

        legend_elements = [
            mpatches.Patch(facecolor=colors[i], alpha=0.6, label=f'Scale {i} ({scales[i]}px)')
            for i in range(min(len(scales), 3))
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()

        image_dir = Path(output_dir) / f"image_{b}"
        image_dir.mkdir(parents=True, exist_ok=True)
        save_path = image_dir / "original_with_tubes.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize multi-scale tube batches")

    parser.add_argument("--train_csv",   type=str, required=True)
    parser.add_argument("--root_dir",    type=str, default="")
    parser.add_argument("--predict_model", action="store_true")

    parser.add_argument("--num_tubes",    type=int,          default=8)
    parser.add_argument("--scales",       type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--target_size",  type=int,          default=128)
    parser.add_argument("--num_views",    type=int,          default=2)
    parser.add_argument("--min_image_size", type=int,        default=256)
    parser.add_argument("--max_image_size", type=int,        default=2048)

    parser.add_argument("--batch_size",          type=int, default=2)
    parser.add_argument("--num_samples",         type=int, default=2)
    parser.add_argument("--num_tubes_per_sample", type=int, default=4)
    parser.add_argument("--output_dir",           type=str, default="visualizations")
    parser.add_argument("--no-normalize",         action="store_true")

    args = parser.parse_args()
    use_normalize = not args.no_normalize

    print(f"\n{'='*60}")
    print(f"MULTI-SCALE TUBE VISUALIZATION")
    print(f"{'='*60}\n")

    dataset = MultiScaleTubeDataset(
        data_path=args.train_csv,
        predict_model=args.predict_model,
        root_dir=args.root_dir,
    )
    print(f"Dataset size: {len(dataset)}")

    collator = MultiScaleTubeCollator(
        num_tubes=args.num_tubes,
        scales=args.scales,
        target_size=args.target_size,
        num_views=args.num_views,
        normalize=use_normalize,
        min_image_size=args.min_image_size,
        max_image_size=args.max_image_size,
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )

    print("Loading batch...")
    batch = next(iter(dataloader))

    # 1) Original images with tube regions
    plot_original_images_with_tube_centers(
        batch, num_samples=args.num_samples, output_dir=args.output_dir,
    )

    # 2) Full representations (RGB | Residual | Wavelet) per tube
    print("\nGenerating tube representations...")
    plot_batch_summary(
        batch,
        num_samples=args.num_samples,
        num_tubes_per_sample=args.num_tubes_per_sample,
        output_dir=args.output_dir,
        denorm=use_normalize,
    )

    # 3) Scale comparison for first image, first tube
    print("\nGenerating scale comparison...")
    plot_scale_comparison(
        batch, image_idx=0, tube_idx=0, view_idx=0,
        output_dir=args.output_dir, denorm=use_normalize,
    )

    # 4) View comparison for first image, first tube, middle scale
    mid_scale = len(args.scales) // 2
    print("\nGenerating view comparison...")
    plot_view_comparison(
        batch, image_idx=0, tube_idx=0, scale_idx=mid_scale,
        output_dir=args.output_dir, denorm=use_normalize,
    )

    print(f"\n{'='*60}")
    print(f"Visualizations saved to: {args.output_dir}/")
    print(f"{'='*60}\n")

    viz_dir = Path(args.output_dir)
    for img_dir in sorted(d for d in viz_dir.iterdir() if d.is_dir() and d.name.startswith("image_")):
        print(f"  {img_dir.name}/")
        for f in sorted(img_dir.glob("*.png")):
            print(f"    - {f.name}")


if __name__ == "__main__":
    main()
