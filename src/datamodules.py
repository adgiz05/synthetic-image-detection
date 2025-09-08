import pytorch_lightning as pl
import torch
from src.datasets import SelfContrastivePretrainingDataset
from src.collators import ViewCollator

class SelfContrastivePretrainingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=200, num_workers=4, dataset_config={}):
        super(SelfContrastivePretrainingDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_config = dataset_config

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SelfContrastivePretrainingDataset(**self.dataset_config)
            self.val_dataset = SelfContrastivePretrainingDataset(split='test', n_views=1, randaug=False)
        
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