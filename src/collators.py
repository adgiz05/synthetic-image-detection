import torch
from transformers import AutoProcessor

class ImageCollator:
    def __init__(self, model_id='google/vit-base-patch16-224-in21k'):
        self.model_id = model_id
        
    def __call__(self, batch):
        images = [item['image'] for item in batch]
        labels = [item['label'] for item in batch]
        images = torch.stack(images, dim=0)            # [B, C, H, W]
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels

class ViewCollator:
    def __call__(self, batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Arrange images along a new views channel [B, N, C, H, W]
        images = torch.stack([torch.stack(imgs, dim=0) for imgs in images], dim=0)

        labels = torch.stack(labels, dim=0) # [B, num_labels]
        return images, labels