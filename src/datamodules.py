import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from src.datasets import *
from src.collators import *

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=200, num_workers=4, train_dataset_config={}, val_dataset_config={}):
        super(ImageDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset_config = train_dataset_config
        self.val_dataset_config = val_dataset_config

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageDataset(split='train', **self.train_dataset_config)
            self.val_dataset = ImageDataset(split='val', **self.val_dataset_config)
            # No augmentation for test just patching
            self.test_dataset = ImageDataset(split='test', size=self.train_dataset_config['size'], augmentation_config={'transformations': False}) 
        self.collate_fn = ImageCollator()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            # prefetch_factor=4,
            shuffle=True,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            # prefetch_factor=4,
            shuffle=False,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn
        )

class SelfContrastivePretrainingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=200, num_workers=4, dataset_config={}):
        super(SelfContrastivePretrainingDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_config = dataset_config

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SelfContrastivePretrainingDataset(split='train', **self.dataset_config)
            self.val_dataset = SelfContrastivePretrainingDataset(split='val', **self.dataset_config) # TODO: Validation without augmentation    
        self.collate_fn = ViewCollator()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn
        )

class NoiseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=16, dataset_config={}, cached=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_config = dataset_config
        self.cached = cached

    def setup(self, stage=None):
        dataset_cls = NoiseDatasetPT if self.cached else NoiseDataset
        self.train_dataset = dataset_cls(split='train', **self.dataset_config)
        self.val_dataset = dataset_cls(split='val', **self.dataset_config)
        if dataset_cls == NoiseDataset:
            self.test_dataset = dataset_cls(split='test', **self.dataset_config)

    @staticmethod
    def collate_crop_pad(batch, patch_size=128):
        tensors, labels = zip(*batch)  # lista de tensores y labels
        out_tensors = []
        for t in tensors:
            _, h, w = t.shape
            # padding si hace falta
            pad_h = max(patch_size - h, 0)
            pad_w = max(patch_size - w, 0)
            if pad_h > 0 or pad_w > 0:
                t = F.pad(t, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
                _, h, w = t.shape
            # crop centrado
            top = (h - patch_size) // 2
            left = (w - patch_size) // 2
            t = t[:, top:top+patch_size, left:left+patch_size]
            out_tensors.append(t)
        return torch.stack(out_tensors), torch.tensor(labels)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda b: self.collate_crop_pad(b, patch_size=self.dataset_config.get('patch_size', 128)),
            shuffle=True,
            pin_memory=True,                 # use page-locked memory to speed up H2D copies
            persistent_workers=True,         # keep workers alive between epochs
            prefetch_factor=2,               # each worker prefetches N batches
            drop_last=True,                  # keeps batch shapes stable and avoids stragglers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda b: self.collate_crop_pad(b, patch_size=self.dataset_config.get('patch_size', 128)),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda b: self.collate_crop_pad(b, patch_size=self.dataset_config.get('patch_size', 128)),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
