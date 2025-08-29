from torchvision import transforms as T

IMAGINET_MEAN = (0.485, 0.456, 0.406)
IMAGINET_STD = (0.229, 0.224, 0.225)
IMAGINET_SIZE = 256

class TwoViewsTransform:
    """Create two views of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def default_imaginet_transform(randaug=False, multiview=False):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGINET_MEAN, std=IMAGINET_STD)
    ])

    if randaug:
        transform.transforms.insert(0, T.RandAugment(num_ops=2, magnitude=9))
    
    if multiview: # Create two views of the same image for contrastive learning
        transform = TwoViewsTransform(transform)
    
    return transform