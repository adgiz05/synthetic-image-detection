import torch
import io
import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from transformers import AutoProcessor
from src.utils import extract_patches_tensor, extract_patches_tensor_auto

class ImageCollator:
    def __init__(self, model_id='google/vit-base-patch16-224-in21k'):
        self.model_id = model_id
        
    def __call__(self, batch):
        images = [item['image'] for item in batch]
        labels = [item['label'] for item in batch]
        images = torch.stack(images, dim=0)            # [B, C, H, W]
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels

class CRPatchedCollator:
    """
    Collator that simulates repeated social-media-like degradation.
    A geometric chain of steps is applied:
        while random() < step_continue_prob AND steps < max_steps:
            optional resize
            optional compression

    After degradation, a random patch of size `patch_size` is extracted.

    A per-image mask is returned:  
        0 = no transform  
        1 = compression  
        2 = resize  
        3 = both
    """

    def __init__(self,
                 step_continue_prob: float = 0.5,
                 max_steps: int = 10,
                 compression_prob: float = 0.5,
                 resize_prob: float = 0.5,
                 jpeg_ratio: float = 0.5,
                 min_quality: int = 70,
                 max_quality: int = 90,
                 min_resize_ratio: float = 0.8,
                 max_resize_ratio: float = 1.0,
                 patch_size: int = 224,
                 return_original: bool = False):
        self.step_continue_prob = step_continue_prob
        self.max_steps = max_steps

        self.compression_prob = compression_prob
        self.resize_prob = resize_prob
        self.jpeg_ratio = jpeg_ratio
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.min_resize_ratio = min_resize_ratio
        self.max_resize_ratio = max_resize_ratio

        self.patch_size = patch_size
        self.return_original = return_original

    def degrade_once(self, img: Image.Image):
        """
        Apply a single degradation step.
        Returns:
            new_img, transform_mask
        Where mask is:
            0 = no op
            1 = compression
            2 = resize
            3 = both
        """
        mask = 0

        # Resize
        if random.random() < self.resize_prob:
            w, h = img.size
            ratio = random.uniform(self.min_resize_ratio, self.max_resize_ratio)
            new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))
            img = F.resize(img, (new_h, new_w), interpolation=Image.BILINEAR)
            mask |= 2  # mark resize

        # Compression
        if random.random() < self.compression_prob:
            fmt = "JPEG" if random.random() < self.jpeg_ratio else "WEBP"
            quality = random.randint(self.min_quality, self.max_quality)
            buf = io.BytesIO()
            img.save(buf, format=fmt, quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            mask |= 1  # mark compression

        return img, mask

    def random_crop(self, img: Image.Image) -> Image.Image:
        """Extract a random patch of size patch_size x patch_size."""
        w, h = img.size
        patch = self.patch_size

        # Upsample if needed
        if w < patch or h < patch:
            scale = max(patch / w, patch / h)
            new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
            img = F.resize(img, (new_h, new_w), interpolation=Image.BILINEAR)
            w, h = img.size

        # Random crop
        left = random.randint(0, w - patch)
        top = random.randint(0, h - patch)

        return F.crop(img, top, left, patch, patch)

    def __call__(self, batch):
        """
        Args:
            batch: list of (image, label)
        Returns:
            dict with:
              - images
              - labels
              - transforms (transform code per image)
              - originals (optional)
        """
        images = [item['image'] for item in batch]
        labels = [item['label'] for item in batch]
        degraded_imgs = []
        originals = []
        transforms = []

        for img in images:
            # Convert tensor → PIL
            if isinstance(img, torch.Tensor):
                img = F.to_pil_image(img)

            if self.return_original:
                originals.append(F.to_tensor(img))

            # Track transform mask for this image
            total_mask = 0

            # Geometric loop of degradation
            steps = 0
            while True:
                img, m = self.degrade_once(img)
                total_mask |= m  # accumulate mask
                steps += 1

                # break if stopping conditions met
                if steps >= self.max_steps or random.random() > self.step_continue_prob:
                    break

            # Final random patch
            img = self.random_crop(img)

            degraded_imgs.append(F.to_tensor(img))
            transforms.append(total_mask)

        degraded_imgs = torch.stack(degraded_imgs)
        labels = torch.tensor(labels, dtype=torch.long)
        transforms = torch.tensor(transforms, dtype=torch.long)
        output = {
            "images": degraded_imgs,
            "labels": labels,
            "transforms": transforms
        }
        if self.return_original:
            output["originals"] = torch.stack(originals)

        return output

class FullImageCollator:
    def __init__(self, patch_size=224, stride=None, max_patches=32):
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches = max_patches

    def __call__(self, batch):
        seqs, coords, labels, lengths = [], [], [], []
        for item in batch:
            img = item['image']  # [C,H,W]
            # convert img to tensor if needed
            if not isinstance(img, torch.Tensor):
                img = F.to_tensor(img)
            if self.stride is None:
                patches, c = extract_patches_tensor_auto(img, self.patch_size, self.max_patches)
            else:
                patches, c = extract_patches_tensor(img, self.patch_size, self.stride)
            seqs.append(patches); coords.append(c)
            labels.append(item['label'])
            lengths.append(patches.shape[0])

        B = len(batch); Nmax = max(lengths)
        C, P = seqs[0].shape[1], seqs[0].shape[2]
        images = torch.zeros(B, Nmax, C, P, P)
        coord_t = torch.zeros(B, Nmax, 2)
        mask = torch.zeros(B, Nmax, dtype=torch.bool)

        for i, (p, c, l) in enumerate(zip(seqs, coords, lengths)):
            images[i, :l] = p
            coord_t[i, :l] = c
            mask[i, :l] = True

        labels = torch.tensor(labels, dtype=torch.long)
        return (images, mask, coord_t), labels


class ViewCollator:
    def __call__(self, batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Arrange images along a new views channel [B, N, C, H, W]
        images = torch.stack([torch.stack(imgs, dim=0) for imgs in images], dim=0)

        labels = torch.stack(labels, dim=0) # [B, num_labels]
        return images, labels