"""Dataset classes for full-image classification."""

import os
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data


class FullImageDataset(torch.utils.data.Dataset):
    """
    Dataset for full-image classification from CSV.
    
    Supports:
    - Binary classification (synthetic vs real)
    - Optional model prediction auxiliary task
    - Optional benchmark information
    """
    
    def __init__(
        self, 
        data_path: str, 
        predict_model: bool = False, 
        return_benchmark: bool = False,
        root_dir: str = ""
    ):
        data = pd.read_csv(data_path)

        if "image_path" not in data.columns:
            raise ValueError("CSV must contain 'image_path'.")

        self.image_paths = data["image_path"].tolist()
        self.root_dir = root_dir
        
        # Main binary label: synthetic vs non-synthetic (assumed int 0/1)
        self.has_labels = "label" in data.columns
        if self.has_labels:
            self.labels = data["label"].astype(int).tolist()
        else:
            self.labels = [-1] * len(self.image_paths)

        self.predict_model = predict_model
        self.return_benchmark = return_benchmark

        # Optional model label head
        if self.predict_model:
            if "model" not in data.columns:
                raise ValueError("predict_model=True but 'model' column not found in CSV")
            # Factorize model column into integer class indices (deterministic: sorted unique)
            self.model_label_raw = data["model"].astype(str).tolist()
            self.model_label_names = sorted(data["model"].dropna().unique().tolist())
            _label_to_idx = {name: i for i, name in enumerate(self.model_label_names)}
            self.model_labels = [_label_to_idx.get(str(m), 0) for m in self.model_label_raw]

        # Optional benchmark field
        self.has_benchmark = "benchmark" in data.columns
        if self.has_benchmark:
            self.benchmarks = data["benchmark"].astype(str).tolist()
        else:
            self.benchmarks = None

    def __len__(self) -> int:
        return len(self.image_paths)

    def _blank_image_fallback(self, h: int = 512, w: int = 512) -> Image.Image:
        """Return a blank RGB image if something goes wrong when loading."""
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def _resolve_path(self, p: str) -> str:
        """Resolve relative paths using root_dir if provided."""
        # If path is absolute, use it as-is
        if os.path.isabs(p):
            return p
        
        # If no root_dir specified, use path as-is
        if not self.root_dir:
            return p
        
        # If path already starts with root_dir, don't duplicate
        # Handle both with and without trailing slash
        root_normalized = os.path.normpath(self.root_dir)
        path_normalized = os.path.normpath(p)
        
        # Check if path already contains the root_dir at the start
        if path_normalized.startswith(root_normalized + os.sep) or path_normalized.startswith(root_normalized):
            # Path already includes root_dir, use as-is
            return p
        
        # Otherwise, join root_dir with path
        return os.path.join(self.root_dir, p)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_path = self.image_paths[idx]
        path = self._resolve_path(raw_path)
        
        is_fallback = False
        load_error = ""
        
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logging.warning(f"Failed to load image '{path}': {e}. Using blank fallback.")
            img = self._blank_image_fallback()
            is_fallback = True
            load_error = repr(e)

        label = self.labels[idx]

        sample = {
            "image": img, 
            "label": label, 
            "path": raw_path,
            "abs_path": path,
            "is_fallback": is_fallback,
            "load_error": load_error,
        }
        
        if self.predict_model and hasattr(self, 'model_labels'):
            sample["model_label"] = self.model_labels[idx]
            
        if self.return_benchmark and self.benchmarks is not None:
            sample["benchmark"] = self.benchmarks[idx]

        return sample


class MultiScaleTubeDataset(torch.utils.data.Dataset):
    """
    Dataset for multi-scale tube-based contrastive learning.
    
    Each sample returns the full image and metadata. The collator is responsible
    for extracting multi-scale tubes (patches at different scales centered at
    the same spatial location).
    
    Supports:
    - Binary classification (synthetic vs real)
    - Optional model/generator prediction auxiliary task
    """
    
    def __init__(
        self,
        data_path: str,
        predict_model: bool = False,
        return_benchmark: bool = False,
        root_dir: str = ""
    ):
        """
        Args:
            data_path: Path to CSV file with columns: image_path, label, [model], [benchmark]
            predict_model: If True, expects 'model' column for generator classification
            return_benchmark: If True, returns benchmark column if present
            root_dir: Root directory to prepend to relative paths
        """
        data = pd.read_csv(data_path)

        if "image_path" not in data.columns:
            raise ValueError("CSV must contain 'image_path'.")

        self.image_paths = data["image_path"].tolist()
        self.root_dir = root_dir
        self.return_benchmark = return_benchmark

        # Main binary label: synthetic vs real (0 or 1)
        self.has_labels = "label" in data.columns
        if self.has_labels:
            self.labels = data["label"].astype(int).tolist()
        else:
            self.labels = [-1] * len(self.image_paths)

        self.predict_model = predict_model

        # Optional model/generator label
        if self.predict_model:
            if "model" not in data.columns:
                raise ValueError("predict_model=True but 'model' column not found in CSV")
            # Factorize model column into integer class indices
            self.model_label_raw = data["model"].astype(str).tolist()
            self.model_label_names = sorted(data["model"].dropna().unique().tolist())
            _label_to_idx = {name: i for i, name in enumerate(self.model_label_names)}
            self.model_labels = [_label_to_idx.get(str(m), 0) for m in self.model_label_raw]

        # Optional benchmark column
        if "benchmark" in data.columns:
            self.benchmarks = data["benchmark"].tolist()
        else:
            self.benchmarks = None

    def __len__(self) -> int:
        return len(self.image_paths)

    def _blank_image_fallback(self, h: int = 512, w: int = 512) -> Image.Image:
        """Return a blank RGB image if loading fails."""
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def _resolve_path(self, p: str) -> str:
        """Resolve relative paths using root_dir if provided."""
        # If path is absolute, use it as-is
        if os.path.isabs(p):
            return p
        
        # If no root_dir specified, use path as-is
        if not self.root_dir:
            return p
        
        # If path already starts with root_dir, don't duplicate
        # Handle both with and without trailing slash
        root_normalized = os.path.normpath(self.root_dir)
        path_normalized = os.path.normpath(p)
        
        # Check if path already contains the root_dir at the start
        if path_normalized.startswith(root_normalized + os.sep) or path_normalized.startswith(root_normalized):
            # Path already includes root_dir, use as-is
            return p
        
        # Otherwise, join root_dir with path
        return os.path.join(self.root_dir, p)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_path = self.image_paths[idx]
        path = self._resolve_path(raw_path)
        
        is_fallback = False
        load_error = ""
        
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logging.warning(f"Failed to load image '{path}': {e}. Using blank fallback.")
            img = self._blank_image_fallback()
            is_fallback = True
            load_error = repr(e)

        label = self.labels[idx]

        sample = {
            "image": img,  # Full PIL Image - collator will extract tubes
            "label": label,
            "path": raw_path,
            "abs_path": path,
            "is_fallback": is_fallback,
            "load_error": load_error,
        }

        if self.predict_model and hasattr(self, 'model_labels'):
            sample["model_label"] = self.model_labels[idx]

        if self.return_benchmark and self.benchmarks is not None:
            sample["benchmark"] = self.benchmarks[idx]

        return sample
