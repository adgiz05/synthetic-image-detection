from PIL import Image as PILImage
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

IMAGINET_MEAN = (0.485, 0.456, 0.406)
IMAGINET_STD = (0.229, 0.224, 0.225)
IMAGINET_SIZE = 256

class NViewsTransform:
    """Create N views of the same image"""
    def __init__(self, pre_transform=None, n_views=1, randaug=False):
        self.pre_transform = pre_transform
        self.n_views = n_views
    
        self.transform = T.Compose([
            *([T.RandAugment(num_ops=2, magnitude=9)] if randaug else []),
            T.ToTensor(),
            T.Normalize(mean=IMAGINET_MEAN, std=IMAGINET_STD)
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