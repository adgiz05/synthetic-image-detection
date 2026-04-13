"""
GPU-accelerated preprocessing for residual and wavelet computation.

This module moves heavy preprocessing from CPU (collator) to GPU (model forward),
significantly reducing GPU idle time by overlapping CPU data loading with GPU compute.

Usage:
    preproc = GPUPreprocessor(target_size=96)
    # In model forward:
    residual, wavelet = preproc(rgb_patches)  # rgb_patches: [B, 3, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPUPreprocessor(nn.Module):
    """
    Batched GPU preprocessing for computing residual and wavelet representations.

    Replaces per-patch CPU operations in the collator with efficient batched GPU ops:
    - Residual: r = patch - gaussian_blur(patch) using depthwise conv2d
    - Wavelet: Haar DWT level-1 on luminance → LH/HL/HH bands

    Args:
        target_size: output spatial size (for wavelet upsampling)
        blur_kernel_size: Gaussian blur kernel size (must be odd)
        blur_sigma: Gaussian blur standard deviation
    """

    def __init__(
        self,
        target_size: int = 96,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0,
    ):
        super().__init__()
        self.target_size = target_size

        # Pre-compute Gaussian kernel for residual computation
        kernel = self._create_gaussian_kernel(blur_kernel_size, blur_sigma)
        # Register as buffer (not a parameter, but moves with .to(device))
        # Shape: [3, 1, K, K] for depthwise conv (groups=3)
        self.register_buffer('gaussian_kernel', kernel.repeat(3, 1, 1, 1))
        self.blur_padding = blur_kernel_size // 2

        # Haar wavelet kernels for DWT level-1
        # Row filters: [1, 1]/2 (low-pass), [1, -1]/2 (high-pass)
        haar_row_lo = torch.tensor([[[[1., 1.]]]], dtype=torch.float32) / 2
        haar_row_hi = torch.tensor([[[[1., -1.]]]], dtype=torch.float32) / 2
        # Col filters
        haar_col_lo = torch.tensor([[[[1.], [1.]]]], dtype=torch.float32) / 2
        haar_col_hi = torch.tensor([[[[1.], [-1.]]]], dtype=torch.float32) / 2

        self.register_buffer('haar_row_lo', haar_row_lo)
        self.register_buffer('haar_row_hi', haar_row_hi)
        self.register_buffer('haar_col_lo', haar_col_lo)
        self.register_buffer('haar_col_hi', haar_col_hi)

        # Luminance weights (ITU-R BT.601)
        self.register_buffer('lum_weights', torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1))

    @staticmethod
    def _create_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = g.outer(g)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

    def compute_residual(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Compute high-frequency residual: r = patch - gaussian_blur(patch).

        Args:
            patches: [B, 3, H, W] RGB patches in [0, 1]

        Returns:
            residual: [B, 3, H, W] zero-centered residual
        """
        # Depthwise separable Gaussian blur (groups=3 for per-channel)
        blurred = F.conv2d(
            patches,
            self.gaussian_kernel,
            padding=self.blur_padding,
            groups=3
        )
        return patches - blurred

    def compute_wavelet(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Haar DWT level-1 on luminance; returns LH/HL/HH stacked as 3 channels.

        Args:
            patches: [B, 3, H, W] RGB patches in [0, 1]

        Returns:
            wavelet: [B, 3, target_size, target_size] normalized wavelet bands
        """
        B = patches.shape[0]

        # Convert to luminance: [B, 1, H, W]
        gray = (patches * self.lum_weights).sum(dim=1, keepdim=True)

        # Row-wise filtering (stride along width)
        row_lo = F.conv2d(gray, self.haar_row_lo, stride=(1, 2))  # [B, 1, H, W//2]
        row_hi = F.conv2d(gray, self.haar_row_hi, stride=(1, 2))

        # Column-wise filtering (stride along height)
        LH = F.conv2d(row_lo, self.haar_col_hi, stride=(2, 1))  # [B, 1, H//2, W//2]
        HL = F.conv2d(row_hi, self.haar_col_lo, stride=(2, 1))
        HH = F.conv2d(row_hi, self.haar_col_hi, stride=(2, 1))

        # Stack bands: [B, 3, H//2, W//2]
        bands = torch.cat([LH, HL, HH], dim=1)

        # Normalize per-sample (batch-wise std normalization)
        # Reshape to [B, -1] for computing std per sample
        bands_flat = bands.view(B, -1)
        std = bands_flat.std(dim=1, keepdim=True).clamp(min=1e-8)
        bands_flat = bands_flat / std
        bands = bands_flat.view_as(bands)

        # Clamp outliers
        bands = torch.clamp(bands, -3.0, 3.0)

        # Resize to target_size
        if bands.shape[-1] != self.target_size:
            bands = F.interpolate(
                bands,
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False,
            )

        return bands

    def forward(self, patches: torch.Tensor) -> tuple:
        """
        Compute both residual and wavelet representations.

        Args:
            patches: [B, 3, H, W] RGB patches in [0, 1] (NOT ImageNet-normalized)

        Returns:
            residual: [B, 3, H, W] high-frequency residual
            wavelet: [B, 3, target_size, target_size] wavelet bands
        """
        residual = self.compute_residual(patches)
        wavelet = self.compute_wavelet(patches)
        return residual, wavelet


class GPUAugmentor(nn.Module):
    """
    GPU-accelerated augmentation for creating views.

    Replaces CPU-bound PIL operations (JPEG, blur, noise) with GPU equivalents.
    Note: True JPEG compression is not differentiable; we use approximations.

    Args:
        jpeg_prob: probability of applying JPEG-like degradation
        jpeg_quality_range: (min, max) quality (higher = less compression)
        blur_prob: probability of applying Gaussian blur
        blur_sigma_range: (min, max) blur sigma
        noise_prob: probability of adding Gaussian noise
        noise_std_range: (min, max) noise standard deviation
    """

    def __init__(
        self,
        jpeg_prob: float = 0.5,
        jpeg_quality_range: tuple = (70, 95),
        blur_prob: float = 0.2,
        blur_sigma_range: tuple = (0.5, 2.0),
        noise_prob: float = 0.3,
        noise_std_range: tuple = (0.01, 0.05),
    ):
        super().__init__()
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_prob = blur_prob
        self.blur_sigma_range = blur_sigma_range
        self.noise_prob = noise_prob
        self.noise_std_range = noise_std_range

        # Pre-compute blur kernels for different sigmas (discretized)
        self._blur_kernels = {}

    def _get_blur_kernel(self, sigma: float, device: torch.device) -> torch.Tensor:
        """Get or create a Gaussian blur kernel for the given sigma."""
        sigma_key = round(sigma * 10)  # Discretize to 0.1 precision
        if sigma_key not in self._blur_kernels:
            size = int(6 * sigma + 1) | 1  # Ensure odd
            size = max(3, min(size, 11))  # Clamp to [3, 11]
            coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            kernel = g.outer(g)
            kernel = kernel / kernel.sum()
            self._blur_kernels[sigma_key] = kernel.unsqueeze(0).unsqueeze(0)
        return self._blur_kernels[sigma_key].to(device)

    def apply_jpeg_approx(self, x: torch.Tensor, quality: int) -> torch.Tensor:
        """
        Approximate JPEG compression using DCT-domain quantization.

        This is a differentiable approximation that mimics block artifacts.
        For training robustness, not for exact JPEG simulation.
        """
        # Simple approximation: low-pass filter + quantization noise
        # More severe for lower quality
        noise_scale = (100 - quality) / 500.0  # 0.0 to 0.06
        noise = torch.randn_like(x) * noise_scale

        # Add slight block artifacts by operating on 8x8 blocks
        B, C, H, W = x.shape
        if H >= 8 and W >= 8:
            # Downsample and upsample to simulate block artifacts
            scale = max(1, (100 - quality) // 20)  # 1-5 based on quality
            if scale > 1:
                small = F.avg_pool2d(x, kernel_size=scale, stride=scale)
                x = F.interpolate(small, size=(H, W), mode='nearest')

        return torch.clamp(x + noise, 0, 1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to create a view.

        Args:
            patches: [B, 3, H, W] RGB patches in [0, 1]

        Returns:
            augmented: [B, 3, H, W] augmented patches
        """
        x = patches
        B = x.shape[0]
        device = x.device

        # Per-sample random decisions (for diversity within batch)
        # Note: For simplicity, we apply same aug to whole batch here
        # For more diversity, could use per-sample masks

        # JPEG approximation
        if torch.rand(1).item() < self.jpeg_prob:
            quality = torch.randint(
                self.jpeg_quality_range[0],
                self.jpeg_quality_range[1] + 1,
                (1,)
            ).item()
            x = self.apply_jpeg_approx(x, quality)

        # Gaussian blur
        if torch.rand(1).item() < self.blur_prob:
            sigma = torch.empty(1).uniform_(*self.blur_sigma_range).item()
            kernel = self._get_blur_kernel(sigma, device)
            padding = kernel.shape[-1] // 2
            # Apply per-channel
            kernel_3ch = kernel.repeat(3, 1, 1, 1)
            x = F.conv2d(x, kernel_3ch, padding=padding, groups=3)

        # Gaussian noise
        if torch.rand(1).item() < self.noise_prob:
            std = torch.empty(1).uniform_(*self.noise_std_range).item()
            noise = torch.randn_like(x) * std
            x = torch.clamp(x + noise, 0, 1)

        return x
