import torch

class ViewCollator:
    def __call__(self, batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Arrange images along a new views channel [B, N, C, H, W]
        images = torch.stack([torch.stack(imgs, dim=0) for imgs in images], dim=0)

        labels = torch.stack(labels, dim=0) # [B, num_labels]
        return images, labels