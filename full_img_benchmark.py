import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from src.datasets import FullImageDataset
from src.modules import FullImageModule
from src.collators import FullImageCollator
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
ckpt_path = "imaginet-full-img-cls/1z4fdw5t/checkpoints/epoch=0-step=17010.ckpt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 32
num_workers = 16

# ============================================================
# MODEL
# ============================================================
print(f"→ Loading model from {ckpt_path} on {device} ...")

model = FullImageModule.load_from_checkpoint(
    ckpt_path, map_location=device
)
model.eval()

# ============================================================
# DATA
# ============================================================
test_dataset = FullImageDataset(
    split="test",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
    shuffle=False,
    collate_fn=FullImageCollator(),
)

# ============================================================
# TEST LOOP
# ============================================================
all_paths, all_sources, all_labels = [], [], []
all_preds, all_confs = [], []

@torch.no_grad()
def run_inference():
    for batch_idx, batch in tqdm(enumerate(test_loader), desc="Inference"):
        (images, mask, coords), labels = batch  # 👈 Correcto

        images = images.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_size = labels.shape[0]

        # Recuperar índices de dataset
        start = batch_idx * batch_size
        end = start + batch_size
        df_batch = test_dataset.data.iloc[start:end]

        sources = df_batch.get("source", ["unknown"] * batch_size)
        paths = df_batch["image_path"].tolist()

        # 👇 Ajuste: el modelo espera (images, mask, coords), no solo images
        outputs = model((images, mask, coords), labels)
        logits = outputs["synthetic_logits"]
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().tolist()
        confs = probs.max(dim=1).values.cpu().tolist()

        all_paths.extend(paths)
        all_sources.extend(sources)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds)
        all_confs.extend(confs)


run_inference()

# ============================================================
# METRICS BY ['source', 'label']
# ============================================================
df_results = pd.DataFrame({
    "image_path": all_paths,
    "source": all_sources,
    "true_label": all_labels,
    "pred_label": all_preds,
    "confidence": all_confs,
})


df_results['label'] = df_results['true_label'].apply(lambda x: x[0])

# opcional: guardar CSV
df_results.to_csv("results/full_img/test_predictions_by_source.csv", index=False)
print("\n💾 Guardado: results/full_img/test_predictions_by_source.csv")

# global metrics
# acc_global = (df_results["label"] == df_results["pred_label"]).mean()
# print(f"\n✅ Global accuracy: {acc_global:.4f}")

# # grouped metrics
# grouped = (
#     df_results
#     .groupby(["source", "label"])
#     .apply(lambda g: (g["label"] == g["pred_label"]).mean())
#     .reset_index(name="accuracy")
#     .sort_values(["source", "label"])
# )
# print("\n📊 Accuracy por [source, label]:")
# print(grouped)

# grouped.to_csv("results/datasetV2_filtered_fullimg.csv", index=False)


