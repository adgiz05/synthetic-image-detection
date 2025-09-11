from .utils import NViewsTransform
from .augmentations import PatchAugmentation

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 split: str = 'train', # Dataset split (corresponding to a CSV file in data/)
                 task: str = 'all', # Task to perform: 'all', 'label', 'content_type', 'model', 'specific_model'
                 augmentation: str = 'patched', # Augmentation to apply: 'patched' or 'none'
                 size: int = 224, # Size of the image or patch
                 augmentation_config: dict = {}, # Additional config for augmentation
                 return_residual: bool = False, # Whether to return the residual (original - augmented)
                 return_original: bool = False # Whether to return the original image
                 ):
        self.data = pd.read_csv(f'data/{split}.csv')

        # Define label with respect to the task
        if task == 'all':
            self.task = ['label', 'content_type', 'model', 'specific_model']
        elif task in ['label', 'content_type', 'model', 'specific_model']:
            self.task = [task]
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Augmentations
        match augmentation:
            case 'patched':
                self.augmentation = PatchAugmentation(size=size, **augmentation_config)
            case _:
                self.augmentation = lambda x: x # No augmentation

        self.return_residual = return_residual # TODO: try adding the residual
        self.return_original = return_original 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # IMAGE LOADING
        original = np.asarray(Image.open(row['image_path']).convert('RGB')) # Load as a numpy array

        # IMAGE AUGMENTATION
        image = self.augmentation(original)

        # LABEL FORMAT
        label = [row[t] for t in self.task]

        return {
            **({'original' : original} if self.return_original else {}),
            'image' : image,
            'label' : label
        }

class SelfContrastivePretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', task='all', augmentation='patched', size=96, n_views=1, randaug=False):
        self.data = pd.read_csv(f'data/{split}.csv')

        # Define label with respect to the task
        if task == 'all':
            self.task = ['label', 'content_type', 'model', 'specific_model']
        elif task in ['label', 'content_type', 'model', 'specific_model']:
            self.task = [task]
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Augmentations
        match augmentation:
            case 'patched':
                self.augmentation = PatchAugmentation(size=size)
            case _:
                self.augmentation = lambda x: x # No augmentation

        # Transformation      
        self.transform = NViewsTransform(
            pre_transform=self.augmentation,
            n_views=n_views,
            randaug=randaug
        )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # IMAGE LOADING
        image = np.asarray(Image.open(row['image_path']).convert('RGB')) # Load as a numpy array

        # IMAGE TRANSFORMATION
        image = self.transform(image)

        # LABEL FORMAT
        label = torch.tensor([row[t] for t in self.task], dtype=torch.long)

        return image, label