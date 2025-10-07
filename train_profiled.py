#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-file, end-to-end, profileado para detectar bottlenecks de entrenamiento.

Requisitos:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install transformers pandas pillow albumentations opencv-python tensorboard

Datos esperados:
  data/train.csv, data/val.csv con columnas:
    image_path (str), label (int: 0 real, 1 sintético),
    content_type (int), model (int), specific_model (int)
  (Si alguna columna no existe, el script te avisa.)

Trazas generadas:
  - ./logs/profile_tb/  -> para TensorBoard: tensorboard --logdir logs/profile_tb
  - ./logs/profile_json/trace_*.json -> Chrome tracing (chrome://tracing)
"""

import os, sys, time, json, argparse, random, math, warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from transformers import AutoModel

try:
    import albumentations as A
    import cv2
except Exception as e:
    A = None
    cv2 = None
    print("[WARN] Albumentations/OpenCV no disponibles, se usarán no-ops para augmentations.")

# ===================== Utilidades =====================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def available_int(col, default=0):
    try:
        return int(col)
    except:
        return default

# ===================== Dataset =====================

class PatchAugmentation:
    """Subset razonable de tus augs para no hacerlas excesivamente pesadas.
       Ajusta a gusto para estresar CPU si lo necesitas."""
    def __init__(self, size=224, enable=True):
        self.size = size
        self.enable = enable
        if A is not None and enable:
            self.tf = A.Compose([
                A.OneOf([
                    A.Downscale(scale_min=0.5, scale_max=0.9, p=1.0),
                    A.NoOp(p=1.0),
                ], p=0.3),

                A.OneOf([
                    A.Resize(size, size, interpolation=cv2.INTER_NEAREST),
                    A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
                    A.Resize(size, size, interpolation=cv2.INTER_CUBIC),
                    A.Resize(size, size, interpolation=cv2.INTER_AREA),
                ], p=0.3),

                # Solo JPEG (WebP puede ser caro)
                A.ImageCompression(quality_lower=40, quality_upper=95, compression_type=0, p=0.3),

                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(var_limit=(2.0, 6.0), p=1.0),
                    A.NoOp(p=1.0),
                ], p=0.1),

                A.OneOf([
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=1.0),
                    A.ToGray(p=1.0),
                    A.NoOp(p=1.0),
                ], p=0.1),

                A.PadIfNeeded(size, size),
                A.RandomCrop(size, size),
            ], p=1.0)
        else:
            self.tf = None

    def __call__(self, img_uint8_hwc):
        if self.tf is None:
            # Sencillo resize+center crop en numpy si no hay albumentations
            pil = Image.fromarray(img_uint8_hwc)
            pil = pil.resize((self.size, self.size), Image.BILINEAR)
            return np.asarray(pil)
        return self.tf(image=img_uint8_hwc)["image"]

class ImageCSV(Dataset):
    def __init__(self, csv_path, size=224, use_aug=True):
        self.df = pd.read_csv(csv_path)
        self.size = size
        self.aug = PatchAugmentation(size=size, enable=use_aug)

        # Post-transform en TENSORES (se ejecuta en worker)
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Campos esperados; si no están, generamos algo por defecto
        self.cols = {
            "label": "label" if "label" in self.df.columns else None,
            "content_type": "content_type" if "content_type" in self.df.columns else None,
            "model": "model" if "model" in self.df.columns else None,
            "specific_model": "specific_model" if "specific_model" in self.df.columns else None,
        }
        missing = [k for k, v in self.cols.items() if v is None]
        if missing:
            print(f"[WARN] Columnas ausentes en {csv_path}: {missing}. Se rellenarán con 0.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        try:
            img = Image.open(path).convert("RGB")
            img = np.asarray(img)  # uint8 HxWxC
            img = self.aug(img)
        except Exception as e:
            print(f"[WARN] No se pudo cargar {path}: {e}")
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        x = self.to_tensor(Image.fromarray(img))  # CHW float

        # Etiquetas
        label      = available_int(row[self.cols["label"]]) if self.cols["label"] else 0
        content    = available_int(row[self.cols["content_type"]]) if self.cols["content_type"] else 0
        model      = available_int(row[self.cols["model"]]) if self.cols["model"] else 0
        spec_model = available_int(row[self.cols["specific_model"]]) if self.cols["specific_model"] else 0

        y_vec = torch.tensor([label, content, model, spec_model], dtype=torch.long)
        return {"image": x, "label": y_vec}

def collate_fn(batch):
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return imgs, labels

# ===================== Modelo y pérdidas =====================

NUM_GENERATORS = 8
SYN_IDX = 0
GEN_IDX = 2

class CLSHead(nn.Module):
    def __init__(self, in_features, out_features, kind="mlp", p_drop=0.1):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(in_features, out_features)
        else:
            self.net = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p_drop),
                nn.Linear(in_features // 2, out_features),
            )
    def forward(self, x):
        return self.net(x)

class DualSyntheticLoss(nn.Module):
    """CE sintético-real + CE del generador (solo en muestras sintéticas)."""
    def __init__(self, synthetic_weight=1.0, model_weight=1.0):
        super().__init__()
        self.sw = synthetic_weight
        self.mw = model_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits_syn, logits_gen, labels):
        syn_labels = labels[:, SYN_IDX]  # [B]
        loss_syn = self.sw * self.ce(logits_syn, syn_labels)
        mask = syn_labels == 1
        if mask.any():
            loss_gen = self.mw * self.ce(logits_gen[mask], labels[mask, GEN_IDX])
        else:
            loss_gen = logits_syn.new_zeros(())
        return loss_syn, loss_gen

class ViTClassifier(nn.Module):
    def __init__(self, model_id="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        hs = self.backbone.config.hidden_size
        self.syn_head = CLSHead(hs, 2, kind="mlp")
        self.gen_head = CLSHead(hs, NUM_GENERATORS, kind="mlp")
        self.crit = DualSyntheticLoss()

    def forward(self, x, labels=None):
        with torch.autograd.profiler.record_function("forward/backbone"):
            out = self.backbone(pixel_values=x).pooler_output  # [B, H]
        with torch.autograd.profiler.record_function("forward/heads"):
            syn_logits = self.syn_head(out)
            gen_logits = self.gen_head(out)
        loss_syn = loss_gen = None
        if labels is not None:
            with torch.autograd.profiler.record_function("loss/compute"):
                loss_syn, loss_gen = self.crit(syn_logits, gen_logits, labels)
        return {"syn": syn_logits, "gen": gen_logits, "feat": out, "loss_syn": loss_syn, "loss_gen": loss_gen}

# ===================== Entrenamiento + Profiling =====================

def worker_init_fn(_):
    # Limitar hilos por worker para no pelear con OpenMP/BLAS
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    if cv2 is not None:
        try:
            cv2.setNumThreads(0)
        except:
            pass

def build_loader(csv_path, batch_size, num_workers, prefetch_factor, pin_memory, persistent_workers, shuffle, aug):
    ds = ImageCSV(csv_path, size=224, use_aug=aug)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

def train_one_epoch(model, loader, optimizer, device, scaler=None, log_every=10):
    model.train()
    data_times = []
    comp_times = []
    end = time.perf_counter()

    for it, (x, y) in enumerate(loader):
        data_t = time.perf_counter() - end

        with torch.autograd.profiler.record_function("h2d_copy"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        torch.cuda.nvtx.range_push("compute_step")
        c_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            out = model(x, labels=y)
            loss = out["loss_syn"] + out["loss_gen"]
            loss.backward()
            optimizer.step()
        else:
            with torch.autograd.profiler.record_function("autocast"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x, labels=y)
                    loss = out["loss_syn"] + out["loss_gen"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        comp_t = time.perf_counter() - c_start
        torch.cuda.nvtx.range_pop()

        data_times.append(data_t)
        comp_times.append(comp_t)

        if (it + 1) % log_every == 0:
            mem = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"[it {it+1:04d}] data_time={data_t*1000:.1f}ms | compute_time={comp_t*1000:.1f}ms | loss={loss.item():.4f} | max_mem={mem:.0f}MB")

        end = time.perf_counter()

    return float(np.mean(data_times)), float(np.mean(comp_times))

def validate_one_epoch(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x, labels=y)
            loss = out["loss_syn"] + out["loss_gen"]
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else math.nan

# ===================== Profiler helpers =====================

def make_profiler(tb_dir, json_dir, activities, record_shapes=True, profile_memory=True, with_stacks=True):
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    step_holder = {"i": 0}

    def on_trace_ready(p):
        # TensorBoard
        p.export_chrome_trace(os.path.join(json_dir, f"trace_{step_holder['i']:04d}.json"))
        p.export_stacks(os.path.join(json_dir, f"stacks_{step_holder['i']:04d}.txt"), "self_cpu_time_total")
        p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
        step_holder["i"] += 1

    prof = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=on_trace_ready,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stacks,
        with_flops=False,
    )
    return prof

# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="data/reduced_splits/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/reduced_splits/val.csv")
    parser.add_argument("--model_id", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--no_aug", action="store_true", default=False)
    parser.add_argument("--tb_dir", type=str, default="logs/profile_tb")
    parser.add_argument("--json_dir", type=str, default="logs/profile_json")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataloaders
    train_loader = build_loader(
        args.train_csv, args.batch_size, args.num_workers,
        args.prefetch_factor, args.pin_memory, args.persistent_workers,
        shuffle=True, aug=not args.no_aug
    )
    val_loader = build_loader(
        args.val_csv, args.batch_size, args.num_workers,
        max(2, args.prefetch_factor//2), args.pin_memory, args.persistent_workers,
        shuffle=False, aug=False
    )

    # Modelo
    model = ViTClassifier(model_id=args.model_id)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # usamos bf16 autocast, no fp16

    # Profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    profiler = make_profiler(args.tb_dir, args.json_dir, activities)

    # Loop
    print("==> Comenzando entrenamiento con profiling...")
    with profiler:
        for epoch in range(args.epochs):
            print(f"\n[Epoch {epoch+1}/{args.epochs}]")
            profiler.step()
            dt, ct = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler)
            profiler.step()
            val_loss = validate_one_epoch(model, val_loader, device)
            profiler.step()
            print(f"Epoch {epoch+1}: mean_data_time={dt*1000:.1f}ms | mean_compute_time={ct*1000:.1f}ms | val_loss={val_loss:.4f}")

    # Resumen de ops
    print("\n==> Resumen de operaciones (key_averages):")
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as p:
        # Ejecutamos una pasada para recoger estadísticas
        model.eval()
        with torch.no_grad():
            for it, (x, y) in enumerate(val_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = model(x, labels=y)
                if it >= 5:
                    break
    print(p.key_averages().table(sort_by="self_cuda_time_total" if device.type=="cuda" else "self_cpu_time_total", row_limit=25))

    # Pistas de diagnóstico rápidas
    print("\n==> Pistas rápidas:")
    print("- Si 'DataLoader::<list>' o 'DataLoader::<batch>' domina en CPU, el cuello está en el input pipeline.")
    print("- Si ves mucho tiempo en 'aten::to' o 'copy_' HtoD, revisa pin_memory y tamaño de batch.")
    print("- Si 'forward/backbone' es pequeño vs data_time, GPU se queda hambrienta.")
    print("- Abre las trazas en TensorBoard (logs/profile_tb) o en Chrome Tracing (logs/profile_json).")

if __name__ == "__main__":
    main()
