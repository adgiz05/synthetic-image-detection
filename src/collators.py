"""Collators for training and testing with optional data augmentation."""

import io
import random
from typing import Tuple, List, Dict, Any

import numpy as np
import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from .constants import IMAGENET_MEAN, IMAGENET_STD
from .patch_utils import extract_patches_tensor_auto


class TrainFullImageDegradePatchCollator:
    """
    Training collator that:
      1) Applies a geometric chain of degradations (resize + compression)
      2) Extracts multiple patches from the degraded image using an automatic stride
      3) Returns:
           - batched patches padded to max length
           - per-image label
           - per-image transform mask (0..3)
           - per-patch coords and attention mask
           - optionally original (pre-degradation) tensors

    Transform mask (per original image):
        0 = no transform
        1 = compression applied at least once
        2 = resize applied at least once
        3 = both compression and resize were applied
    """

    def __init__(
        self,
        # degradation chain
        step_continue_prob: float = 0.5,
        max_steps: int = 10,
        compression_prob: float = 0.5,
        resize_prob: float = 0.5,
        jpeg_ratio: float = 0.5,
        min_quality: int = 70,
        max_quality: int = 90,
        min_resize_ratio: float = 0.8,
        max_resize_ratio: float = 1.0,
        # patch extraction
        patch_size: int = 224,
        max_patches: int = 32,
        # options
        return_original: bool = False,
        normalize: bool = True,
    ):
        # Degradation params
        self.step_continue_prob = step_continue_prob
        self.max_steps = max_steps
        self.compression_prob = compression_prob
        self.resize_prob = resize_prob
        self.jpeg_ratio = jpeg_ratio
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.min_resize_ratio = min_resize_ratio
        self.max_resize_ratio = max_resize_ratio

        # Patch params
        self.patch_size = patch_size
        self.max_patches = max_patches

        self.return_original = return_original

        # Transforms for tensor conversion / normalization
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) if normalize else None

    # ------------- Degradation functions -------------

    def degrade_once(self, img: Image.Image) -> Tuple[Image.Image, int]:
        """
        Apply a single degradation step.

        Returns:
            new_img, transform_mask

        Where transform_mask bits are:
            1 = compression happened
            2 = resize happened
        """
        mask = 0

        # Resize
        if random.random() < self.resize_prob:
            w, h = img.size
            ratio = random.uniform(self.min_resize_ratio, self.max_resize_ratio)
            new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))
            img = TF.resize(img, (new_h, new_w))
            mask |= 2  # resize bit

        # Compression
        if random.random() < self.compression_prob:
            fmt = "JPEG" if random.random() < self.jpeg_ratio else "WEBP"
            
            # WebP has a limit of 16383 pixels per dimension
            # Fall back to JPEG if image exceeds this limit
            if fmt == "WEBP":
                w, h = img.size
                if w > 16383 or h > 16383:
                    fmt = "JPEG"
            
            quality = random.randint(self.min_quality, self.max_quality)
            buf = io.BytesIO()
            img.save(buf, format=fmt, quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            mask |= 1  # compression bit

        return img, mask

    def degrade_image_chain(self, img: Image.Image) -> Tuple[Image.Image, int]:
        """
        Apply a geometric chain of degradation steps.
        With probability (1 - step_continue_prob) no degradation is applied at all
        (transform mask = 0), allowing the model to learn the no-transform class.

        Returns:
            degraded_img, total_mask (0..3)
        """
        total_mask = 0
        steps = 0

        while steps < self.max_steps and random.random() < self.step_continue_prob:
            img, m = self.degrade_once(img)
            total_mask |= m
            steps += 1

        return img, total_mask

    def __call__(self, batch):
        """
        Args:
            batch: list of dicts {"image": PIL.Image or Tensor, "label": int}

        Returns:
            output dict with:
                - images:     [B, Nmax, C, P, P]
                - coords:     [B, Nmax, 2]
                - attn_mask:  [B, Nmax] (True where a patch exists)
                - labels:     [B]
                - transforms: [B] (0..3)
                - originals:  [B, C, H, W] (optional)
        """
        seqs = []          # list of [Ni, C, P, P]
        coords_list = []   # list of [Ni, 2]
        lengths = []       # list of Ni
        labels = []        # list of labels
        transforms = []    # per-image transform code
        originals = []     # optional pre-degradation tensors
        model_labels = []  # optional model labels

        for item in batch:
            img = item["image"]
            label = item["label"]

            # Optional model label
            if "model_label" in item:
                model_labels.append(item["model_label"])

            # Ensure we start from a PIL image for degradation
            if isinstance(img, torch.Tensor):
                img = TF.to_pil_image(img)

            # Optionally store the original (pre-degradation) tensor
            if self.return_original:
                orig_t = self.to_tensor(img)
                if self.normalize is not None:
                    orig_t = self.normalize(orig_t)
                originals.append(orig_t)

            # Apply degradation chain
            degraded_img, t_mask = self.degrade_image_chain(img)
            transforms.append(t_mask)

            # Convert degraded image to tensor
            t = self.to_tensor(degraded_img)
            if self.normalize is not None:
                t = self.normalize(t)

            # Extract patches
            patches, c = extract_patches_tensor_auto(
                t,
                self.patch_size,
                self.max_patches
            )

            seqs.append(patches)
            coords_list.append(c)
            lengths.append(patches.shape[0])
            labels.append(label)

        # Pad to batch with max sequence length
        B = len(batch)
        Nmax = max(lengths)
        C, P = seqs[0].shape[1], seqs[0].shape[2]

        images = torch.zeros(B, Nmax, C, P, P, dtype=seqs[0].dtype)
        coord_t = torch.zeros(B, Nmax, 2, dtype=coords_list[0].dtype)
        attn_mask = torch.zeros(B, Nmax, dtype=torch.bool)

        for i, (p, c, l) in enumerate(zip(seqs, coords_list, lengths)):
            images[i, :l] = p
            coord_t[i, :l] = c
            attn_mask[i, :l] = True

        # Convert labels / transforms to tensors (assuming integer labels)
        labels = torch.as_tensor(labels, dtype=torch.long)
        transforms = torch.as_tensor(transforms, dtype=torch.long)

        output = {
            "images": images,        # [B, Nmax, C, P, P]
            "coords": coord_t,       # [B, Nmax, 2]
            "attn_mask": attn_mask,  # [B, Nmax]
            "labels": labels,        # [B]
            "transforms": transforms # [B]
        }

        # Optional original full images
        if self.return_original and len(originals) > 0:
            output["originals"] = torch.stack(originals, dim=0)  # [B, C, H, W]

        # Optional model labels
        if len(model_labels) > 0:
            output["model_label"] = torch.as_tensor(model_labels, dtype=torch.long)  # [B]

        return output


class ValFullImagePatchCollator:
    """
    Validation/Test collator that:
      1) DOES NOT apply degradations (no resize/compression noise)
      2) Only normalizes and tiles the image into patches.
      3) Returns the same structure as the train collator, but
         with transform codes set to 0 (no transform).
    """

    def __init__(
        self,
        patch_size: int = 224,
        max_patches: int = 32,
        normalize: bool = True,
        return_original: bool = False,
        return_benchmark: bool = False,
    ):
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.return_original = return_original
        self.return_benchmark = return_benchmark
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) if normalize else None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Args:
            batch: list of dicts {"image": PIL.Image or Tensor, "label": int, ...}

        Returns:
            output dict with:
                - images:     [B, Nmax, C, P, P]
                - coords:     [B, Nmax, 2]
                - attn_mask:  [B, Nmax]
                - labels:     [B]
                - transforms: [B] (all zeros)
                - originals:  [B, C, H, W] (optional)
                - benchmarks: list (optional)
                - image_paths: list (optional, for testing)
                - abs_paths: list (optional, for testing)
                - is_fallback: list (optional, for testing)
                - load_errors: list (optional, for testing)
        """
        seqs = []
        coords_list = []
        lengths = []
        labels = []
        transforms = []
        originals = []
        benchmarks = []
        model_labels = []
        image_paths = []
        abs_paths = []
        is_fallback = []
        load_errors = []

        for item in batch:
            img = item["image"]
            label = item["label"]

            if self.return_benchmark and "benchmark" in item:
                benchmarks.append(item["benchmark"])

            # Optional model label
            if "model_label" in item:
                model_labels.append(item["model_label"])

            # Optional testing fields
            if "path" in item:
                image_paths.append(item["path"])
            if "abs_path" in item:
                abs_paths.append(item["abs_path"])
            if "is_fallback" in item:
                is_fallback.append(item["is_fallback"])
            if "load_error" in item:
                load_errors.append(item["load_error"])

            # Ensure PIL input
            if isinstance(img, torch.Tensor):
                img = TF.to_pil_image(img)

            # Optional original tensor (non-degraded)
            if self.return_original:
                orig_t = self.to_tensor(img)
                if self.normalize is not None:
                    orig_t = self.normalize(orig_t)
                originals.append(orig_t)

            # Convert to tensor and normalize
            t = self.to_tensor(img)
            if self.normalize is not None:
                t = self.normalize(t)

            # Extract patches (same auto logic as train, but no degradation)
            patches, c = extract_patches_tensor_auto(
                t,
                self.patch_size,
                self.max_patches
            )

            seqs.append(patches)
            coords_list.append(c)
            lengths.append(patches.shape[0])
            labels.append(label)
            transforms.append(0)  # no transform for validation

        B = len(batch)
        Nmax = max(lengths)
        C, P = seqs[0].shape[1], seqs[0].shape[2]

        images = torch.zeros(B, Nmax, C, P, P, dtype=seqs[0].dtype)
        coord_t = torch.zeros(B, Nmax, 2, dtype=coords_list[0].dtype)
        attn_mask = torch.zeros(B, Nmax, dtype=torch.bool)

        for i, (p, c, l) in enumerate(zip(seqs, coords_list, lengths)):
            images[i, :l] = p
            coord_t[i, :l] = c
            attn_mask[i, :l] = True

        labels = torch.as_tensor(labels, dtype=torch.long)
        transforms = torch.as_tensor(transforms, dtype=torch.long)

        output = {
            "images": images,
            "coords": coord_t,
            "attn_mask": attn_mask,
            "labels": labels,
            "transforms": transforms,
        }

        if self.return_original and len(originals) > 0:
            output["originals"] = torch.stack(originals, dim=0)

        if self.return_benchmark and benchmarks:
            output["benchmarks"] = benchmarks

        if len(model_labels) > 0:
            output["model_label"] = torch.as_tensor(model_labels, dtype=torch.long)

        # Optional testing fields
        if image_paths:
            output["image_paths"] = image_paths
        if abs_paths:
            output["abs_paths"] = abs_paths
        if is_fallback:
            output["is_fallback"] = is_fallback
        if load_errors:
            output["load_errors"] = load_errors

        return output


class MultiScaleTubeCollator:
    """
    Collator for multi-scale tube contrastive learning.
    
    For each image:
      1) Selects N tube centers (spatial locations)
      2) For each center, extracts K patches at different scales (multi-scale tube)
      3) For each patch in the tube, creates V augmented views with controlled degradations
      
    The intuition:
      - Same spatial location, different scales → scale invariance
      - Augmented views → robustness to degradations while preserving forensic traces
      
    Returns:
      - tubes: [B, N_tubes, K_scales, V_views, C, P, P]
      - tube_centers: [B, N_tubes, 2] - normalized (cy, cx)
      - labels: [B]
      - model_labels: [B] (optional)
    """
    
    def __init__(
        self,
        # Tube configuration
        num_tubes: int = 8,
        scales: List[int] = [64, 128, 256],
        target_size: int = 128,
        num_views: int = 2,
        # Degradation params for views
        jpeg_prob: float = 0.5,
        jpeg_quality_range: Tuple[int, int] = (70, 95),
        resize_prob: float = 0.3,
        resize_range: Tuple[float, float] = (0.8, 1.2),
        blur_prob: float = 0.2,
        blur_sigma_range: Tuple[float, float] = (0.5, 2.0),
        sharpen_prob: float = 0.2,
        sharpen_strength_range: Tuple[float, float] = (0.5, 2.0),
        noise_prob: float = 0.3,
        noise_std_range: Tuple[float, float] = (0.01, 0.05),
        # Options
        normalize: bool = True,
        min_image_size: int = 256,  # minimum size to extract largest scale
        max_image_size: int = 2048,  # maximum size - resize if exceeded
    ):
        self.num_tubes = num_tubes
        self.scales = sorted(scales)  # [small -> large]
        self.target_size = target_size
        self.num_views = num_views
        
        # Degradation params
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.resize_prob = resize_prob
        self.resize_range = resize_range
        self.blur_prob = blur_prob
        self.blur_sigma_range = blur_sigma_range
        self.sharpen_prob = sharpen_prob
        self.sharpen_strength_range = sharpen_strength_range
        self.noise_prob = noise_prob
        self.noise_std_range = noise_std_range
        
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) if normalize else None
    
    def _apply_degradations(self, img: Image.Image) -> Image.Image:
        """
        Apply a single random degradation to create a view.
        
        We apply controlled degradations that preserve forensic structure
        but add realistic variations:
        - JPEG/WebP compression
        - Slight resize variations
        - Mild blur
        - Mild sharpen
        - Light Gaussian noise
        """
        # JPEG compression
        if random.random() < self.jpeg_prob:
            quality = random.randint(*self.jpeg_quality_range)
            fmt = "JPEG" if random.random() < 0.7 else "WEBP"
            
            # WebP size limit
            w, h = img.size
            if fmt == "WEBP" and (w > 16383 or h > 16383):
                fmt = "JPEG"
            
            buf = io.BytesIO()
            img.save(buf, format=fmt, quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        
        # Resize variation
        if random.random() < self.resize_prob:
            w, h = img.size
            ratio = random.uniform(*self.resize_range)
            new_w, new_h = max(8, int(w * ratio)), max(8, int(h * ratio))
            img = TF.resize(img, (new_h, new_w))
            # Resize back to original size
            img = TF.resize(img, (h, w))
        
        # Blur
        if random.random() < self.blur_prob:
            sigma = random.uniform(*self.blur_sigma_range)
            img = TF.gaussian_blur(img, kernel_size=5, sigma=sigma)
        
        # Sharpen (via unsharpen mask approximation)
        if random.random() < self.sharpen_prob:
            strength = random.uniform(*self.sharpen_strength_range)
            # Simple sharpen: img + strength * (img - blur(img))
            blurred = TF.gaussian_blur(img, kernel_size=5, sigma=1.0)
            img_t = self.to_tensor(img)
            blur_t = self.to_tensor(blurred)
            sharp_t = img_t + strength * (img_t - blur_t)
            sharp_t = torch.clamp(sharp_t, 0, 1)
            img = TF.to_pil_image(sharp_t)
        
        # Additive Gaussian noise
        if random.random() < self.noise_prob:
            img_t = self.to_tensor(img)
            noise_std = random.uniform(*self.noise_std_range)
            noise = torch.randn_like(img_t) * noise_std
            noisy_t = torch.clamp(img_t + noise, 0, 1)
            img = TF.to_pil_image(noisy_t)
        
        return img
    
    def _extract_tube(self, img: Image.Image, center_y: float, center_x: float) -> List[torch.Tensor]:
        """
        Extract patches at multiple scales centered at (center_y, center_x).
        
        Args:
            img: PIL Image
            center_y: normalized center y in [0, 1]
            center_x: normalized center x in [0, 1]
            
        Returns:
            List of K tensors [C, target_size, target_size], one per scale
        """
        w, h = img.size
        cy_pix = int(center_y * h)
        cx_pix = int(center_x * w)
        
        scale_patches = []
        
        for scale in self.scales:
            # Compute crop box centered at (cx_pix, cy_pix)
            half = scale // 2
            x1 = max(0, cx_pix - half)
            y1 = max(0, cy_pix - half)
            x2 = min(w, x1 + scale)
            y2 = min(h, y1 + scale)
            
            # Adjust if we hit boundaries
            if x2 - x1 < scale:
                x1 = max(0, x2 - scale)
            if y2 - y1 < scale:
                y1 = max(0, y2 - scale)
            
            # Extract crop
            crop = img.crop((x1, y1, x2, y2))
            
            # Resize to target size (all scales become same size)
            crop_resized = TF.resize(crop, (self.target_size, self.target_size))
            
            # Convert to tensor
            crop_t = self.to_tensor(crop_resized)
            scale_patches.append(crop_t)
        
        return scale_patches
    
    def _compute_information_map(self, img: Image.Image, downsample_factor: int = 4) -> np.ndarray:
        """
        Compute an information/saliency map for the image.
        
        Uses Laplacian magnitude (edge detection) as a proxy for information content.
        Downsampled for efficiency.
        
        Args:
            img: PIL Image
            downsample_factor: factor to downsample for speed
            
        Returns:
            2D numpy array [H', W'] with normalized information scores
        """
        # Convert to grayscale numpy array
        img_gray = np.array(img.convert('L'))
        
        # Downsample for efficiency
        h, w = img_gray.shape
        new_h, new_w = h // downsample_factor, w // downsample_factor
        img_small = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Compute Laplacian magnitude (edges/texture)
        laplacian = cv2.Laplacian(img_small, cv2.CV_64F)
        info_map = np.abs(laplacian)
        
        # Smooth to get regional information
        info_map = cv2.GaussianBlur(info_map, (5, 5), 0)
        
        # Normalize to [0, 1]
        if info_map.max() > 0:
            info_map = info_map / info_map.max()
        
        # Add small baseline to allow sampling everywhere (avoid zeros)
        info_map = info_map * 0.9 + 0.1
        
        return info_map
    
    def _sample_tube_centers(self, img: Image.Image) -> List[Tuple[float, float]]:
        """
        Sample N tube centers from the image with:
        1. Priority to information-rich regions
        2. Overlap control via minimum distance
        
        Returns:
            List of (cy, cx) in normalized coordinates [0, 1]
        """
        w, h = img.size
        max_scale = self.scales[-1]
        
        # Define valid region (avoid edges where largest scale won't fit)
        margin_y = max_scale / (2 * h)
        margin_x = max_scale / (2 * w)
        
        # Compute information map
        info_map = self._compute_information_map(img, downsample_factor=4)
        info_h, info_w = info_map.shape
        
        # Minimum distance between centers (in normalized coords)
        # Set to ~1.5x the largest scale to reduce overlap
        min_distance = 1.5 * max_scale / max(h, w)
        
        centers = []
        max_attempts = self.num_tubes * 20  # avoid infinite loop
        attempts = 0
        
        while len(centers) < self.num_tubes and attempts < max_attempts:
            attempts += 1
            
            # Sample from information map using weighted probability
            # Flatten and normalize to probability distribution
            flat_probs = info_map.flatten()
            flat_probs = flat_probs / flat_probs.sum()
            
            # Sample an index
            idx = np.random.choice(len(flat_probs), p=flat_probs)
            iy, ix = np.unravel_index(idx, (info_h, info_w))
            
            # Convert to normalized coordinates [0, 1]
            cy = (iy + 0.5) / info_h
            cx = (ix + 0.5) / info_w
            
            # Check margins
            if cy < margin_y or cy > 1 - margin_y:
                continue
            if cx < margin_x or cx > 1 - margin_x:
                continue
            
            # Check distance to existing centers
            too_close = False
            for existing_cy, existing_cx in centers:
                dist = np.sqrt((cy - existing_cy)**2 + (cx - existing_cx)**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                centers.append((cy, cx))
                
                # Suppress this region in info_map to avoid sampling nearby again
                # This speeds up convergence
                iy_min = max(0, int(iy - info_h * min_distance / 2))
                iy_max = min(info_h, int(iy + info_h * min_distance / 2))
                ix_min = max(0, int(ix - info_w * min_distance / 2))
                ix_max = min(info_w, int(ix + info_w * min_distance / 2))
                info_map[iy_min:iy_max, ix_min:ix_max] *= 0.1
        
        # If we couldn't get enough centers with constraints, fill with random
        while len(centers) < self.num_tubes:
            cy = random.uniform(margin_y, 1 - margin_y)
            cx = random.uniform(margin_x, 1 - margin_x)
            centers.append((cy, cx))
        
        return centers
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of images into multi-scale tubes with views.
        
        Args:
            batch: List of dicts with keys: "image", "label", ["model_label"]
            
        Returns:
            Dict with:
                - tubes: [B, N_tubes, K_scales, V_views, C, P, P]
                - tube_centers: [B, N_tubes, 2]
                - labels: [B]
                - model_labels: [B] (optional)
        """
        B = len(batch)
        K = len(self.scales)
        V = self.num_views
        N = self.num_tubes
        C = 3
        P = self.target_size
        
        tubes_list = []
        centers_list = []
        labels = []
        model_labels = []
        image_paths = []
        
        for item in batch:
            img = item["image"]
            label = item["label"]
            
            if "model_label" in item:
                model_labels.append(item["model_label"])
            
            # Store absolute path for visualization
            if "abs_path" in item:
                image_paths.append(item["abs_path"])
            
            # Ensure PIL Image
            if isinstance(img, torch.Tensor):
                img = TF.to_pil_image(img)
            
            # Resize image if outside acceptable range
            w, h = img.size
            
            # Too small: upscale to minimum
            if min(w, h) < self.min_image_size:
                scale_factor = self.min_image_size / min(w, h)
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                img = TF.resize(img, (new_h, new_w))
            
            # Too large: downscale to maximum
            elif max(w, h) > self.max_image_size:
                scale_factor = self.max_image_size / max(w, h)
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                img = TF.resize(img, (new_h, new_w))
            
            # Sample tube centers
            centers = self._sample_tube_centers(img)
            centers_list.append(centers)
            
            # For each tube center
            img_tubes = []
            for cy, cx in centers:
                # Extract multi-scale patches (no augmentation yet)
                scale_patches = self._extract_tube(img, cy, cx)
                
                # For each scale, create V augmented views
                tube_views = []
                for scale_t in scale_patches:
                    views = []
                    
                    # First view: original (or lightly augmented)
                    view0 = scale_t
                    if self.normalize is not None:
                        view0 = self.normalize(view0)
                    views.append(view0)
                    
                    # Additional views: apply degradations
                    for _ in range(V - 1):
                        # Convert tensor back to PIL for degradation
                        pil_patch = TF.to_pil_image(scale_t)
                        aug_patch = self._apply_degradations(pil_patch)
                        aug_t = self.to_tensor(aug_patch)
                        if self.normalize is not None:
                            aug_t = self.normalize(aug_t)
                        views.append(aug_t)
                    
                    # Stack views: [V, C, P, P]
                    tube_views.append(torch.stack(views, dim=0))
                
                # Stack scales: [K, V, C, P, P]
                img_tubes.append(torch.stack(tube_views, dim=0))
            
            # Stack tubes: [N, K, V, C, P, P]
            tubes_list.append(torch.stack(img_tubes, dim=0))
            labels.append(label)
        
        # Stack batch: [B, N, K, V, C, P, P]
        tubes = torch.stack(tubes_list, dim=0)
        
        # Convert centers to tensor: [B, N, 2]
        centers_tensor = torch.tensor(centers_list, dtype=torch.float32)
        
        # Convert labels
        labels = torch.tensor(labels, dtype=torch.long)
        
        output = {
            "tubes": tubes,  # [B, N_tubes, K_scales, V_views, C, P, P]
            "tube_centers": centers_tensor,  # [B, N_tubes, 2]
            "labels": labels,  # [B]
            "scales": self.scales,  # List of scale values for reference
        }
        
        if model_labels:
            output["model_labels"] = torch.tensor(model_labels, dtype=torch.long)
        
        if image_paths:
            output["image_paths"] = image_paths
        
        return output
