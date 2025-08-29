from augmentations import default_augmentation
from transformations import default_imaginet_transform

import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000

class ImagiNet(torch.utils.data.Dataset):
    def __init__(self, split='train', task='all', transform=None, augmentation=None):
        self.split = split
        self.data = pd.read_csv(f'data/{split}.csv')

        # Define label with respect to the task
        if task == 'all':
            self.task = ['label', 'content_type', 'model', 'specific_model']
        elif task in ['label', 'content_type', 'model', 'specific_model']:
            self.task = [task]
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Augmentations
        self.augmentation = augmentation if augmentation is not None else default_augmentation()

        # Transformations
        self.transform = transform if transform is not None else default_imaginet_transform()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # IMAGE LOADING
        image = np.asarray(Image.open(row['image_path']).convert('RGB')) # Load as a numpy array

        if self.split in ['train', 'val']: # Training/Validation set
            image = self.augmentation(image=image)['image']

        image = self.transform(image)

        # LABEL FORMAT
        label = torch.tensor([row[t] for t in self.task], dtype=torch.long)

        return image, label