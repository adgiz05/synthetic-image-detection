import time
import torch
from torch.utils.data import DataLoader
import pandas as pd
import cv2
import numpy as np
import random
import torch.nn.functional as F
import pytorch_lightning as pl
import kornia

# === Dataset for profiling ===
class ProfilingDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, patch_size=128, limit=1000):
        self.data = pd.read_csv(csv_file)
        self.patch_size = patch_size
        self.limit = min(limit, len(self.data))

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        t0 = time.perf_counter()

        img_path = self.data.iloc[idx]["image_path"]
        label = self.data.iloc[idx]["label"]

        # measure imread
        t1 = time.perf_counter()
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image {img_path}")
        t2 = time.perf_counter()

        # crop
        h_img, w_img = img.shape
        ph = self.patch_size
        if h_img < ph or w_img < ph:
            img = cv2.resize(img, (max(w_img, ph), max(h_img, ph)))
            h_img, w_img = img.shape

        y = random.randint(0, h_img - ph)
        x = random.randint(0, w_img - ph)
        patch = img[y:y+ph, x:x+ph]
        t3 = time.perf_counter()

        # normalize
        patch = patch.astype(np.float32) / 255.0
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        t4 = time.perf_counter()

        timings = (t2 - t1, t3 - t2, t4 - t3, t4 - t0)
        return patch_tensor, torch.tensor(label, dtype=torch.long), timings

# === Model with separated timings ===
class NoiseClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3, kernel_size=(7,7), sigma=(1.5,1.5)):
        super().__init__()
        self.save_hyperparameters()
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(64 * 16 * 16, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1)
        )

    def preprocess_noise(self, x):
        denoised = kornia.filters.gaussian_blur2d(x, self.kernel_size, self.sigma)
        noise = (x - denoised).abs()
        return noise

    def forward_with_timing(self, x):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Denoising
        x = self.preprocess_noise(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # CNN + FC
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        return x.squeeze(1), (t1 - t0, t2 - t1, t2 - t0)

# === Profiler ===
def profile_pipeline(csv_file, n_batches=10, batch_size=32, num_workers=8, device="cuda"):
    ds = ProfilingDataset(csv_file, patch_size=128, limit=n_batches*batch_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    model = NoiseClassifier().to(device).eval()

    imread_times, crop_times, norm_times, total_getitem = [], [], [], []
    denoise_times, cnn_times, forward_totals, batch_totals = [], [], [], []

    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        x, y, timings_list = batch
        imread_times.extend(timings_list[0].tolist())
        crop_times.extend(timings_list[1].tolist())
        norm_times.extend(timings_list[2].tolist())
        total_getitem.extend(timings_list[3].tolist())

        # model forward
        x = x.to(device)
        with torch.no_grad():
            out, (t_denoise, t_cnn, t_forward) = model.forward_with_timing(x)

        denoise_times.append(t_denoise)
        cnn_times.append(t_cnn)
        forward_totals.append(t_forward)
        batch_totals.append(sum(timings_list[3]).item() + t_forward)

    print("==== Profiling Results ====")
    print(f"[__getitem__] imread     avg: {np.mean(imread_times):.4f}s | max: {np.max(imread_times):.4f}s")
    print(f"[__getitem__] crop       avg: {np.mean(crop_times):.4f}s | max: {np.max(crop_times):.4f}s")
    print(f"[__getitem__] normalize  avg: {np.mean(norm_times):.4f}s | max: {np.max(norm_times):.4f}s")
    print(f"[__getitem__] total      avg: {np.mean(total_getitem):.4f}s | max: {np.max(total_getitem):.4f}s")
    print(f"[model] denoise (blur)   avg: {np.mean(denoise_times):.4f}s | max: {np.max(denoise_times):.4f}s")
    print(f"[model] cnn+fc           avg: {np.mean(cnn_times):.4f}s | max: {np.max(cnn_times):.4f}s")
    print(f"[model] forward total    avg: {np.mean(forward_totals):.4f}s | max: {np.max(forward_totals):.4f}s")
    print(f"[pipeline] batch total   avg: {np.mean(batch_totals):.4f}s | max: {np.max(batch_totals):.4f}s")

if __name__ == "__main__":
    profile_pipeline("data/train.csv", n_batches=100, batch_size=32, num_workers=8)
