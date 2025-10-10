from .utils import NViewsTransform, IMAGENET_MEAN, IMAGENET_STD
from .augmentations import PatchAugmentation

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms as T
import os
from pathlib import Path

import random

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
                 return_original: bool = False, # Whether to return the original image
                 dataset_size: str = 'full' # 'full' or 'reduced' dataset size
                 ):
        # Load data
        match dataset_size:
            case 'full':
                self.data = pd.read_csv(f'data/{split}.csv')
            case 'reduced':
                self.data = pd.read_csv(f'data/reduced_splits/{split}.csv')
            case 'filtered':
                self.data = pd.read_csv(f'data/filtered_splits/{split}.csv')
            case _:
                raise ValueError(f"Unknown dataset_size: {dataset_size}")

        # Define label with respect to the task
        if task == 'all':
            self.task = ['label', 'content_type', 'model', 'specific_model']
        elif task in ['label', 'content_type', 'model', 'specific_model']:
            self.task = [task]
        else:
            raise ValueError(f"Unknown task: {task}")
        
        self.size = size
        # Augmentations
        match augmentation:
            case 'patched':
                self.augmentation = PatchAugmentation(size=size, **augmentation_config)
            case _:
                self.augmentation = lambda x: x # No augmentation

        self.to_tensor = T.Compose([
            T.ToTensor(),  # HWC uint8 -> CHW float [0,1]
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.return_residual = return_residual # TODO: try adding the residual
        self.return_original = return_original 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = row['image_path']

        try:
            original = np.asarray(Image.open(path).convert('RGB'))  # np.uint8 HxWxC
            aug_img = self.augmentation(original)                   # aún np.uint8 HxWxC
        except:
            print(f"[WARN] Could not load image {path}")
            original = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            aug_img = original

        # A tensor normalizado (CHW float), aquí ya en el worker
        pil_img = Image.fromarray(aug_img)
        image_tensor = self.to_tensor(pil_img)                      # torch.float32 [C,H,W]

        # Etiquetas como lista (el collate las apila a tensor)
        label = [row[t] for t in self.task]

        out = {
            'image': image_tensor,
            'label': label
        }

        if self.return_original:
            out['original'] = original

        return out

class FullImageDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', task='all', size=224):
        self.data = pd.read_csv(f'data/full_img_splits/{split}.csv')

        if task == 'all':
            self.task = ['label', 'content_type', 'model', 'specific_model']
        elif task in ['label','content_type','model','specific_model']:
            self.task = [task]
        else:
            raise ValueError(f"Unknown task: {task}")

        # Solo normalización (sin recortar a size; el tiling lo hará el collator)
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = row['image_path']
        try:
            img = np.asarray(Image.open(path).convert('RGB'))
        except:
            h = w = 512
            img = np.zeros((h, w, 3), dtype=np.uint8)
        tensor = self.to_tensor(Image.fromarray(img))  # [C,H,W] float
        label = [row[t] for t in self.task]
        return {'image': tensor, 'label': label}


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

class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', patch_size=128, transform=None):
        self.data = pd.read_csv(f"data/{split}.csv")
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image_path"]
        label = self.data.iloc[idx]["label"]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image {img_path}")

        # random crop
        h_img, w_img = img.shape
        ph = self.patch_size
        if h_img < ph or w_img < ph:
            img = cv2.resize(img, (max(w_img, ph), max(h_img, ph)))
            h_img, w_img = img.shape

        y = random.randint(0, h_img - ph)
        x = random.randint(0, w_img - ph)
        patch = img[y:y+ph, x:x+ph]

        # normalize [0,1]
        patch = patch.astype(np.float32) / 255.0
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)  # (1, H, W)

        if self.transform:
            patch_tensor = self.transform(patch_tensor)

        return patch_tensor, torch.tensor(label, dtype=torch.long)

class NoiseDatasetPT(torch.utils.data.Dataset):
    def __init__(self, split='train', cache_dir='data/noise/images', patch_size=128):
        """
        Dataset de tensores cacheados (.pt), siempre en escala de grises.
        Si la imagen >= patch_size: crop.
        Si la imagen < patch_size en alguna dimensión: pad hasta patch_size.
        """
        self.cache_dir = Path(cache_dir) / split
        self.data = pd.read_csv(f"data/noise/{split}.csv")
        self.split = split
        self.patch_size = patch_size

        self.to_float = T.ConvertImageDtype(torch.float32)
        self.to_gray = T.Grayscale(num_output_channels=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tensor = torch.load(self.cache_dir / (Path(row.image_path).stem + ".pt"))  # [C,H,W], uint8

        tensor = self.to_float(tensor)  # [C,H,W] en float32 [0,1]
        tensor = self.to_gray(tensor)   # [1,H,W]

        return tensor, int(row.label)