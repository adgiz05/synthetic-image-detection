import os
os.environ["HF_HOME"] = "/opt/huggingface/cache"  # Set Hugging Face cache directory

from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

def _save_png(args):
    idx, image, out_path = args
    # Save lossless PNG quickly (English comments as requested)
    image.save(out_path, format="PNG", compress_level=1)
    return idx, out_path

def _prepare_dirs(root):
    images_dir = os.path.join(root, "images")
    os.makedirs(os.path.join(images_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(images_dir, "ai"), exist_ok=True)
    return images_dir

def download_generic_hf(repo_id, split, dataset_path, label_fn=None, num_workers=8, image_key="image", label_key="label"):
    annotations_file = os.path.join(dataset_path, "annotations.csv")
    if os.path.exists(annotations_file):
        print(f"Dataset already prepared at {dataset_path}. Skipping.")
        return

    os.makedirs(dataset_path, exist_ok=True)
    ds = load_dataset(repo_id, split=split)

    images_dir = _prepare_dirs(dataset_path)

    tasks = []
    annotations = []

    for idx, item in enumerate(ds):
        label = item[label_key]
        if label_fn is not None:
            try:
                label = label_fn(label)
            except:
                print(f"Skipping item {idx} due to label processing error: {label}.")
                continue

        subdir = "ai" if label == 1 else "real"
        out_path = os.path.join(images_dir, subdir, f"{idx:06d}.png")

        tasks.append((idx, item[image_key], out_path))
        annotations.append({"image_id": idx, "image_path": out_path, "label": label})

    # Write images in parallel (English comments as requested)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_save_png, t) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Saving {repo_id}"):
            pass

    pd.DataFrame(annotations).to_csv(annotations_file, index=False)

def download_chameleon(dataset_path="data/benchmarks/chameleon"):
    download_generic_hf(
        repo_id="pranav-5644/Chameleon",
        split="test",
        dataset_path=dataset_path,
        label_fn=lambda x: 1 if x == 0 else 0,  # invert label
        num_workers=8,
    )

def download_aiginow(dataset_path="data/benchmarks/aiginow"):
    download_generic_hf(
        repo_id="Gaffeyzz/AIGI-Now",
        split="train",
        dataset_path=dataset_path,
        label_fn=lambda x: int(x.split('/')[-2][0]),  # keep label
        num_workers=8,
        image_key="jpg",
        label_key="__key__",
    )

def download_sotagenerators(dataset_path="data/benchmarks/sotagenerators"):
    download_generic_hf(
        repo_id="julienlucas/midjourney-dalle-sd-nanobananapro-dataset",
        split="test",
        dataset_path=dataset_path,
        label_fn=lambda x: 1 if x == 0 else 0,  # invert label
        num_workers=8,
    )

if __name__ == "__main__":
    download_chameleon()
    download_sotagenerators()
    download_aiginow()
