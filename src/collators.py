import torch
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

class FullImageCollator:
    def __init__(self, patch_size=224, stride=None, max_patches=32):
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches = max_patches

    def __call__(self, batch):
        seqs, coords, labels, lengths = [], [], [], []
        for item in batch:
            img = item['image']  # [C,H,W]
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