"""
Script to visualize multi-scale tube batches.

This script loads a batch from the training data and plots:
- Multi-scale tubes (different scales at same location)
- Augmented views (different degradations of same patch)
- Labels and metadata

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

# Local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets import MultiScaleTubeDataset
from src.collators import MultiScaleTubeCollator
from src.constants import IMAGENET_MEAN, IMAGENET_STD


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalize a tensor with ImageNet stats.
    
    Args:
        tensor: [C, H, W] or [B, C, H, W]
        
    Returns:
        denormalized tensor
    """
    # Clone to avoid modifying original
    tensor = tensor.clone()
    
    # Create tensors with same device and dtype as input
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    
    if tensor.dim() == 4:  # batch
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    # Denormalize: tensor * std + mean
    # This reverses the normalization: (tensor - mean) / std
    denorm_tensor = tensor * std + mean
    
    return denorm_tensor


def tensor_to_image(tensor, denorm=True):
    """
    Convert a tensor to numpy image for plotting.
    
    Args:
        tensor: [C, H, W] tensor
        denorm: whether to denormalize
        
    Returns:
        [H, W, C] numpy array in [0, 1]
    """
    # Move to CPU first if needed
    tensor = tensor.detach().cpu()
    
    if denorm:
        tensor = denormalize(tensor)
    
    # Convert to numpy: [C, H, W] -> [H, W, C]
    img = tensor.permute(1, 2, 0).numpy()
    
    # Clip to valid range
    img = np.clip(img, 0, 1)
    
    return img


def plot_tube(tube, tube_idx, image_idx, label, center, save_path=None, denorm=True):
    """
    Plot a single multi-scale tube with all its views.
    
    Args:
        tube: [K_scales, V_views, C, P, P] tensor
        tube_idx: index of this tube in the image
        image_idx: index of the image in the batch
        label: image label (0=real, 1=synthetic)
        center: (cy, cx) normalized coordinates
        save_path: optional path to save figure
        denorm: whether to denormalize (should match collator's normalize setting)
    """
    K, V, C, P, _ = tube.shape
    
    fig, axes = plt.subplots(K, V, figsize=(V * 3, K * 3))
    if K == 1:
        axes = axes.reshape(1, -1)
    if V == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(
        f"Image {image_idx} | Tube {tube_idx} | "
        f"Label: {'Synthetic' if label == 1 else 'Real'} | "
        f"Center: ({center[0]:.2f}, {center[1]:.2f})",
        fontsize=14, fontweight='bold'
    )
    
    for k in range(K):
        for v in range(V):
            ax = axes[k, v]
            
            # Get patch and convert to image
            patch = tube[k, v]  # [C, P, P]
            img = tensor_to_image(patch, denorm=denorm)
            
            # Plot
            ax.imshow(img)
            ax.axis('off')
            
            # Add title
            if v == 0:
                ax.set_title(f"Scale {k}\nView {v} (original)", fontsize=10)
            else:
                ax.set_title(f"Scale {k}\nView {v} (augmented)", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_batch_summary(batch, num_samples=2, num_tubes_per_sample=4, output_dir="visualizations", denorm=True):
    """
    Plot a summary of the batch showing multiple tubes from multiple images.
    
    Args:
        batch: batch dict from collator
        num_samples: number of images to visualize
        num_tubes_per_sample: number of tubes to show per image
        output_dir: directory to save plots
        denorm: whether to denormalize (should match collator's normalize setting)
    """
    tubes = batch['tubes']  # [B, N, K, V, C, P, P]
    centers = batch['tube_centers']  # [B, N, 2]
    labels = batch['labels']  # [B]
    
    B, N, K, V, C, P, _ = tubes.shape
    
    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"Batch size: {B}")
    print(f"Tubes per image: {N}")
    print(f"Scales per tube: {K}")
    print(f"Views per scale: {V}")
    print(f"Patch size: {P}x{P}")
    print(f"Labels: {labels.tolist()}")
    if 'model_labels' in batch:
        print(f"Model labels: {batch['model_labels'].tolist()}")
    print(f"{'='*60}\n")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot individual tubes
    num_samples = min(num_samples, B)
    for b in range(num_samples):
        label = labels[b].item()
        
        # Create subdirectory for this image
        image_dir = os.path.join(output_dir, f"image_{b}")
        Path(image_dir).mkdir(parents=True, exist_ok=True)
        
        # Select tubes to visualize (evenly spaced)
        tube_indices = np.linspace(0, N-1, num_tubes_per_sample, dtype=int)
        
        for ti, tube_idx in enumerate(tube_indices):
            tube = tubes[b, tube_idx]  # [K, V, C, P, P]
            center = centers[b, tube_idx]  # [2]
            
            save_path = os.path.join(image_dir, f"tube_{tube_idx}.png")
            plot_tube(
                tube, 
                tube_idx=tube_idx,
                image_idx=b,
                label=label,
                center=center.tolist(),
                save_path=save_path,
                denorm=denorm
            )


def plot_scale_comparison(batch, image_idx=0, tube_idx=0, view_idx=0, output_dir="visualizations", denorm=True):
    """
    Plot a comparison of all scales for a single tube and view.
    
    Args:
        batch: batch dict from collator
        image_idx: which image in the batch
        tube_idx: which tube to visualize
        view_idx: which view to visualize
        output_dir: directory to save plots
        denorm: whether to denormalize (should match collator's normalize setting)
    """
    tubes = batch['tubes']  # [B, N, K, V, C, P, P]
    centers = batch['tube_centers']  # [B, N, 2]
    labels = batch['labels']  # [B]
    
    B, N, K, V, C, P, _ = tubes.shape
    
    tube = tubes[image_idx, tube_idx]  # [K, V, C, P, P]
    center = centers[image_idx, tube_idx]  # [2]
    label = labels[image_idx].item()
    
    fig, axes = plt.subplots(1, K, figsize=(K * 4, 4))
    if K == 1:
        axes = [axes]
    
    fig.suptitle(
        f"Scale Comparison | Image {image_idx} | Tube {tube_idx} | View {view_idx}\n"
        f"Label: {'Synthetic' if label == 1 else 'Real'} | "
        f"Center: ({center[0]:.2f}, {center[1]:.2f})",
        fontsize=14, fontweight='bold'
    )
    
    for k in range(K):
        ax = axes[k]
        patch = tube[k, view_idx]  # [C, P, P]
        img = tensor_to_image(patch, denorm=denorm)
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Scale {k}", fontsize=12)
    
    plt.tight_layout()
    
    # Save in image subdirectory
    image_dir = os.path.join(output_dir, f"image_{image_idx}")
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(image_dir, f"scale_comparison_tube{tube_idx}_view{view_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_view_comparison(batch, image_idx=0, tube_idx=0, scale_idx=1, output_dir="visualizations", denorm=True):
    """
    Plot a comparison of all views for a single tube and scale.
    
    Args:
        batch: batch dict from collator
        image_idx: which image in the batch
        tube_idx: which tube to visualize
        scale_idx: which scale to visualize
        output_dir: directory to save plots
        denorm: whether to denormalize (should match collator's normalize setting)
    """
    tubes = batch['tubes']  # [B, N, K, V, C, P, P]
    centers = batch['tube_centers']  # [B, N, 2]
    labels = batch['labels']  # [B]
    
    B, N, K, V, C, P, _ = tubes.shape
    
    tube = tubes[image_idx, tube_idx]  # [K, V, C, P, P]
    center = centers[image_idx, tube_idx]  # [2]
    label = labels[image_idx].item()
    
    fig, axes = plt.subplots(1, V, figsize=(V * 4, 4))
    if V == 1:
        axes = [axes]
    
    fig.suptitle(
        f"View Comparison | Image {image_idx} | Tube {tube_idx} | Scale {scale_idx}\n"
        f"Label: {'Synthetic' if label == 1 else 'Real'} | "
        f"Center: ({center[0]:.2f}, {center[1]:.2f})",
        fontsize=14, fontweight='bold'
    )
    
    for v in range(V):
        ax = axes[v]
        patch = tube[scale_idx, v]  # [C, P, P]
        img = tensor_to_image(patch, denorm=denorm)
        
        ax.imshow(img)
        ax.axis('off')
        if v == 0:
            ax.set_title(f"View {v}\n(original)", fontsize=12)
        else:
            ax.set_title(f"View {v}\n(augmented)", fontsize=12)
    
    plt.tight_layout()
    
    # Save in image subdirectory
    image_dir = os.path.join(output_dir, f"image_{image_idx}")
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(image_dir, f"view_comparison_tube{tube_idx}_scale{scale_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_original_images_with_tube_centers(batch, num_samples=2, output_dir="visualizations", denorm=True):
    """
    Plot original images with tube centers marked.
    
    This shows the full original images and marks where each tube was extracted from.
    
    Args:
        batch: batch dict from collator (must contain 'image_paths')
        num_samples: number of images to visualize
        output_dir: directory to save plots
        denorm: whether images were normalized (affects title, not used directly)
    """
    image_paths = batch.get('image_paths', [])
    centers = batch['tube_centers']  # [B, N, 2] - normalized coordinates (cy, cx)
    labels = batch['labels']  # [B]
    scales = batch.get('scales', [64, 128, 256])  # Get scales if available
    
    if not image_paths:
        print("⚠️  Warning: batch does not contain 'image_paths', cannot plot originals")
        return
    
    B, N, _ = centers.shape
    
    print(f"\nPlotting original images with tube centers...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    num_samples = min(num_samples, B, len(image_paths))
    
    for b in range(num_samples):
        img_path = image_paths[b]
        label = labels[b].item()
        
        # Load original image
        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            H, W = img_array.shape[:2]
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            continue
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(img_array)
        
        # Draw tube centers and extraction regions
        tube_centers = centers[b]  # [N, 2]
        
        for tube_idx in range(N):
            cy_norm, cx_norm = tube_centers[tube_idx].tolist()
            
            # Convert normalized coordinates to pixel coordinates
            cy = cy_norm * H
            cx = cx_norm * W
            
            # Draw a marker at the center
            ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2)
            
            # Draw rectangles for each scale to show extraction region
            # Use alpha to make them semi-transparent
            colors = ['red', 'orange', 'yellow']
            for scale_idx, scale in enumerate(scales[:3]):  # Show up to 3 scales
                half = scale / 2
                # Rectangle: (x, y) is bottom-left corner
                rect = mpatches.Rectangle(
                    (cx - half, cy - half),  # bottom-left
                    scale,  # width
                    scale,  # height
                    color=colors[scale_idx % len(colors)],
                    fill=False, 
                    linewidth=1.5, 
                    alpha=0.6,
                    linestyle='--'
                )
                ax.add_patch(rect)
            
            # Add tube index label
            ax.text(
                cx + 10, cy - 10, 
                f'T{tube_idx}',
                color='white',
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7)
            )
        
        # Add title and legend
        ax.set_title(
            f"Original Image {b} | Label: {'Synthetic' if label == 1 else 'Real'}\n"
            f"Showing {N} tube extraction locations",
            fontsize=14,
            fontweight='bold'
        )
        ax.axis('off')
        
        # Create legend for scales
        legend_elements = [
            mpatches.Patch(facecolor='red', alpha=0.6, label=f'Scale 0 ({scales[0]}px)'),
        ]
        if len(scales) > 1:
            legend_elements.append(
                mpatches.Patch(facecolor='orange', alpha=0.6, label=f'Scale 1 ({scales[1]}px)')
            )
        if len(scales) > 2:
            legend_elements.append(
                mpatches.Patch(facecolor='yellow', alpha=0.6, label=f'Scale 2 ({scales[2]}px)')
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save in image subdirectory
        image_dir = os.path.join(output_dir, f"image_{b}")
        Path(image_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(image_dir, f"original_with_tubes.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize multi-scale tube batches")
    
    # Data arguments
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train CSV")
    parser.add_argument("--root_dir", type=str, default="", help="Root directory for image paths")
    parser.add_argument("--predict_model", action="store_true", help="Load model labels")
    
    # Tube configuration
    parser.add_argument("--num_tubes", type=int, default=8, help="Number of tubes per image")
    parser.add_argument("--scales", type=int, nargs="+", default=[64, 128, 256], help="Scales for multi-scale tubes")
    parser.add_argument("--target_size", type=int, default=128, help="Target size for patches")
    parser.add_argument("--num_views", type=int, default=2, help="Number of augmented views per patch")
    
    # Image size constraints
    parser.add_argument("--min_image_size", type=int, default=256, help="Minimum image size (upscale if smaller)")
    parser.add_argument("--max_image_size", type=int, default=2048, help="Maximum image size (downscale if larger)")
    
    # Visualization options
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of images to visualize")
    parser.add_argument("--num_tubes_per_sample", type=int, default=4, help="Number of tubes to show per image")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory for plots")
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalization (for debugging)")
    
    args = parser.parse_args()
    
    # Determine if we should normalize
    use_normalize = not args.no_normalize
    
    print(f"\n{'='*60}")
    print(f"MULTI-SCALE TUBE VISUALIZATION")
    print(f"{'='*60}\n")
    
    if not use_normalize:
        print("⚠️  WARNING: Normalization is DISABLED (debug mode)")
        print("   Images will not be normalized with ImageNet stats\n")
    
    # Create dataset
    print("Loading dataset...")
    print(f"CSV path: {args.train_csv}")
    print(f"Root dir: '{args.root_dir}' (empty = paths relative to CWD)")
    dataset = MultiScaleTubeDataset(
        data_path=args.train_csv,
        predict_model=args.predict_model,
        root_dir=args.root_dir,
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Create collator
    print("Creating collator...")
    collator = MultiScaleTubeCollator(
        num_tubes=args.num_tubes,
        scales=args.scales,
        target_size=args.target_size,
        num_views=args.num_views,
        normalize=use_normalize,
        min_image_size=args.min_image_size,
        max_image_size=args.max_image_size,
    )
    
    # Create dataloader
    print("Creating dataloader...")
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for reproducibility in visualization
        collate_fn=collator,
    )
    
    # Load a batch
    print("Loading batch...")
    batch = next(iter(dataloader))
    
    # Plot original images with tube centers
    print("\nGenerating original image visualizations...")
    plot_original_images_with_tube_centers(
        batch,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        denorm=use_normalize
    )
    
    # Plot summary
    print("\nGenerating tube visualizations...")
    plot_batch_summary(
        batch, 
        num_samples=args.num_samples,
        num_tubes_per_sample=args.num_tubes_per_sample,
        output_dir=args.output_dir,
        denorm=use_normalize
    )
    
    # Plot scale comparison
    print("\nGenerating scale comparisons...")
    plot_scale_comparison(
        batch, 
        image_idx=0, 
        tube_idx=0, 
        view_idx=0, 
        output_dir=args.output_dir,
        denorm=use_normalize
    )
    
    # Plot view comparison
    print("\nGenerating view comparisons...")
    plot_view_comparison(
        batch, 
        image_idx=0, 
        tube_idx=0, 
        scale_idx=1, 
        output_dir=args.output_dir,
        denorm=use_normalize
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Visualizations saved to: {args.output_dir}/")
    print(f"{'='*60}\n")
    
    print("Directory structure created:")
    viz_dir = Path(args.output_dir)
    
    # List subdirectories (one per image)
    image_dirs = sorted([d for d in viz_dir.iterdir() if d.is_dir() and d.name.startswith("image_")])
    
    if image_dirs:
        for img_dir in image_dirs:
            print(f"\n  📁 {img_dir.name}/")
            files = sorted(img_dir.glob("*.png"))
            for f in files:
                print(f"     - {f.name}")
    else:
        # Fallback: list all PNG files (old behavior)
        for f in sorted(viz_dir.glob("*.png")):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
