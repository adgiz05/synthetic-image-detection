import torch
from transformers import AutoProcessor

class ImageCollator:
    def __init__(self, model_id='google/vit-base-patch16-224-in21k'):
        self.processor = AutoProcessor.from_pretrained(model_id)
    def __call__(self, batch):
        images = [item['image'] for item in batch]
        labels = [item['label'] for item in batch]

        images = self.processor(images=images, return_tensors='pt')['pixel_values']
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