from PIL import Image as PILImage
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import math
import torch
import torch.nn.functional as F
import math
import transformers

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class NViewsTransform:
    """Create N views of the same image"""
    def __init__(self, pre_transform=None, n_views=1, randaug=False, normalize=True, rescale=True):
        self.pre_transform = pre_transform
        self.n_views = n_views
    
        self.transform = T.Compose([
            *([T.RandAugment(num_ops=2, magnitude=9)] if randaug else []),
            T.ToTensor(),
            *([T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)] if normalize else []),
        ])

    def _view(self, image):
        image = PILImage.fromarray(image)
        return self.transform(image)

    def __call__(self, image):
        if self.pre_transform is not None: # There is some augmentation
            image = self.pre_transform(image=image)        
        
        return [self._view(image) for _ in range(self.n_views)]
    
def load_resnet50_imagenet_weights(resnet):
    m_t = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    st = resnet.encoder.state_dict()
    st1 = m_t.state_dict()
    for name, _ in st.items():
        if "selfcon" not in name:
            st[name].copy_(st1[name.replace("shortcut", "downsample")])
    del st1, m_t
    resnet.encoder.load_state_dict(st)
    return resnet

def sliding_window_indices(H, W, patch, stride):
    ys = list(range(0, max(H - patch, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - patch, 0) + 1, stride)) or [0]
    return [(y, x) for y in ys for x in xs]

def _extract_patches_tensor_fixed(x, patch, stride):
    """
    x: [C,H,W] float/torch
    patch: int
    stride: int  (Fijo, NO auto)
    return: patches [N,C,patch,patch], coords [N,2] (centros normalizados y,x)
    """
    C, H, W = x.shape
    coords, patches = [], []
    for (y, x0) in sliding_window_indices(H, W, patch, stride):
        y2, x2 = y + patch, x0 + patch
        yy2, xx2 = min(y2, H), min(x2, W)
        crop = x[:, y:yy2, x0:xx2]
        # pad si toca borde
        pad_h = patch - crop.shape[1]
        pad_w = patch - crop.shape[2]
        if pad_h > 0 or pad_w > 0:
            crop = F.pad(crop, (0, pad_w, 0, pad_h))
        patches.append(crop)
        cy = min(y + patch / 2, H) / max(H, 1)
        cx = min(x0 + patch / 2, W) / max(W, 1)
        coords.append([cy, cx])
    return torch.stack(patches, 0), torch.tensor(coords, dtype=torch.float32)

# def _compute_stride_auto(H, W, patch_size, max_patches=32):
#     L = max(H, W)
#     if L <= patch_size:
#         return patch_size
#     # nº máximo por eje (<= sqrt(max_patches)), forzamos a cubrir en rejilla compacta
#     n_axis = max(1, int(math.floor(math.sqrt(max_patches))))
#     if n_axis <= 1:
#         return patch_size
#     stride = (L - patch_size) / (n_axis - 1)
#     stride = int(max(1, round(stride)))
#     # Por seguridad, no dejar huecos mayores al patch (puedes quitar esta línea si quieres gaps)
#     stride = min(stride, patch_size)
#     return stride
def _compute_stride_auto(H, W, patch_size, max_patches=32):
    import math
    L = max(H, W)
    if L <= patch_size:
        return patch_size

    n_axis = int(math.sqrt(max_patches))
    while True:
        stride = (L - patch_size) / (n_axis - 1) if n_axis > 1 else patch_size
        n_patches = n_axis ** 2
        if n_patches <= max_patches:
            break
        n_axis -= 1
    return int(max(1, round(stride)))


def extract_patches_tensor(x, patch_size, stride):
    """API pública: extracción con stride FIJO (no auto)."""
    return _extract_patches_tensor_fixed(x, patch_size, stride)

def extract_patches_tensor_auto(x, patch_size=224, max_patches=32):
    """API pública: extracción con stride AUTO limitado por max_patches."""
    C, H, W = x.shape
    stride = _compute_stride_auto(H, W, patch_size, max_patches)
    return _extract_patches_tensor_fixed(x, patch_size, stride)


def positional_encoding_2d(coords, d_model):
    """
    coords: [N,2] in [0,1] (y, x). Returns [N, d_model] with sines/cosines.
    """
    assert d_model % 4 == 0, "d_model must be a multiple of 4"
    N = coords.shape[0]
    pe = torch.zeros(N, d_model)
    div_term = torch.exp(torch.arange(0, d_model//2, 2, dtype=torch.float32) * (-math.log(10000.0) / (d_model//2)))
    for i, ax in enumerate([coords[:,0], coords[:,1]]):  # y, x
        pe[:, 2*i* (d_model//4) : (2*i+1)*(d_model//4)] = torch.sin(ax.unsqueeze(1) * div_term)
        pe[:, (2*i+1)*(d_model//4) : (2*i+2)*(d_model//4)] = torch.cos(ax.unsqueeze(1) * div_term)
    return pe

def register_vit_safe_globals():
    vit = transformers.models.vit.modeling_vit
    cfg = transformers.models.vit.configuration_vit
    torch.serialization.add_safe_globals([
        vit.ViTModel, vit.ViTEmbeddings, vit.ViTPatchEmbeddings,
        vit.ViTEncoder, vit.ViTLayer, vit.ViTAttention,
        vit.ViTSelfAttention, vit.ViTSelfOutput,
        vit.ViTIntermediate, vit.ViTOutput, vit.ViTPooler,
        cfg.ViTConfig,
        transformers.activations.GELUActivation,
        torch.nn.modules.dropout.Dropout,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.normalization.LayerNorm,
        torch.nn.modules.container.ModuleList,
        torch.nn.modules.activation.Tanh,
        torch._C._nn.gelu,
    ])

# -------------------------------------------------------------------------
# Debug visualization helpers
# -------------------------------------------------------------------------

def decode_transform_mask(mask_code: int):
    """
    Decode a transform mask into human-readable components.

    Bits:
        1 -> compression
        2 -> resize
    """
    components = []
    if mask_code & 2:
        components.append("resize")
    if mask_code & 1:
        components.append("compression")
    if mask_code == 0:
        components.append("none")
    return components

def denormalize_image(t: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor image that was normalized with IMAGENET_MEAN/STD.
    Expects tensor in [C,H,W].
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=t.dtype, device=t.device).view(3, 1, 1)
    return t * std + mean


def save_tensor_as_image(t: torch.Tensor, path: str, denormalize: bool = True):
    """
    Save a tensor image [C,H,W] to disk as PNG.
    If denormalize=True, apply inverse ImageNet normalization first.
    """
    if denormalize:
        t = denormalize_image(t)

    t = t.clamp(0.0, 1.0)
    # Convert from [C,H,W] to [H,W,C]
    np_img = t.permute(1, 2, 0).cpu().numpy()
    np_img = (np_img * 255).astype(np.uint8)
    img = Image.fromarray(np_img)
    img.save(path)


def make_patch_grid(patches: torch.Tensor,
                    max_patches: int = 16) -> torch.Tensor:
    """
    Build a big grid image from a set of patches.

    Args:
        patches:     [N, C, H, W]
        max_patches: Max number of patches to visualize

    Returns:
        grid: [C, H_grid, W_grid] tensor
    """
    N, C, H, W = patches.shape
    N_use = min(N, max_patches)
    patches = patches[:N_use]

    # Grid size ~ square
    cols = int(math.ceil(math.sqrt(N_use)))
    rows = int(math.ceil(N_use / cols))

    grid = torch.zeros(C, rows * H, cols * W, dtype=patches.dtype)

    for i in range(N_use):
        r = i // cols
        c = i % cols
        grid[:, r * H:(r + 1) * H, c * W:(c + 1) * W] = patches[i]

    return grid


def visualize_batch(batch: dict,
                    out_dir: str,
                    split: str,
                    batch_idx: int,
                    max_images_per_batch: int = 4,
                    max_patches_per_image: int = 16):
    """
    Visualize a batch from the collator and return metadata describing
    what transformations were applied.

    Saves, for each image:
      - A grid of degraded patches.
      - The original image if 'originals' is present in the batch.

    Returns:
        logs: list of dicts with:
              - split
              - batch_idx
              - image_idx
              - path
              - transform_code
              - transformations (list of str)
              - patches_file
              - original_file (optional)
    """
    images = batch["images"]        # [B, N, C, P, P]
    attn_mask = batch["attn_mask"]  # [B, N]
    transforms = batch["transforms"]  # [B]
    originals = batch.get("originals", None)  # [B, C, H, W] if present
    paths = batch.get("paths", [None] * images.shape[0])

    B, N, C, P, P = images.shape

    os.makedirs(out_dir, exist_ok=True)

    logs = []
    num_imgs = min(B, max_images_per_batch)
    for b in range(num_imgs):
        # Optional: save original image if available
        orig_path = None
        if originals is not None:
            orig = originals[b]  # [C, H, W], normalized
            orig_path = os.path.join(
                out_dir,
                f"{split}_b{batch_idx}_i{b}_original.png"
            )
            save_tensor_as_image(orig, orig_path, denormalize=True)

        # Select valid patches using attn_mask
        valid_idx = torch.nonzero(attn_mask[b], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue

        patches_b = images[b, valid_idx]  # [Ni, C, P, P]
        grid = make_patch_grid(patches_b, max_patches=max_patches_per_image)

        t_code = int(transforms[b].item())
        grid_path = os.path.join(
            out_dir,
            f"{split}_b{batch_idx}_i{b}_patches_t{t_code}.png"
        )
        save_tensor_as_image(grid, grid_path, denormalize=True)

        log_entry = {
            "split": split,
            "batch_idx": int(batch_idx),
            "image_idx": int(b),
            "path": paths[b],
            "transform_code": t_code,
            "transformations": decode_transform_mask(t_code),
            "patches_file": os.path.abspath(grid_path),
            "original_file": os.path.abspath(orig_path) if orig_path is not None else None,
        }
        logs.append(log_entry)

    return logs


def debug_visualize_dataloaders(datamodule: FullImageDataModule,
                                output_dir: str,
                                num_batches: int = 1,
                                max_images_per_batch: int = 4,
                                max_patches_per_image: int = 16):
    """
    Iterate over train/val dataloaders and save a few visualizations
    of how patches+degradations look.

    Also writes a debug_transforms.yaml file with the info about
    which transformations were applied to which images.
    """
    # Ensure datasets are ready
    datamodule.setup()

    summary = {"train": [], "val": []}

    # Train loader
    train_loader = datamodule.train_dataloader()
    train_out_dir = os.path.join(output_dir, "debug_viz_train")
    for batch_idx, batch in enumerate(train_loader):
        logs = visualize_batch(
            batch,
            out_dir=train_out_dir,
            split="train",
            batch_idx=batch_idx,
            max_images_per_batch=max_images_per_batch,
            max_patches_per_image=max_patches_per_image,
        )
        summary["train"].extend(logs)
        if batch_idx + 1 >= num_batches:
            break

    # Val loader
    val_loader = datamodule.val_dataloader()
    val_out_dir = os.path.join(output_dir, "debug_viz_val")
    for batch_idx, batch in enumerate(val_loader):
        logs = visualize_batch(
            batch,
            out_dir=val_out_dir,
            split="val",
            batch_idx=batch_idx,
            max_images_per_batch=max_images_per_batch,
            max_patches_per_image=max_patches_per_image,
        )
        summary["val"].extend(logs)
        if batch_idx + 1 >= num_batches:
            break

    # Write YAML summary
    yaml_path = os.path.join(output_dir, "debug_transforms.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(summary, f)

    print(f"[DEBUG] Saved debug visualizations to {train_out_dir} and {val_out_dir}")
    print(f"[DEBUG] Saved transform summary to {yaml_path}")