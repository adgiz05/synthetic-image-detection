from PIL import Image as PILImage
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import math
import torch
import torch.nn.functional as F

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

import math
import torch
import torch.nn.functional as F

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
