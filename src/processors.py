from PIL import Image as PILImage
import torchvision.transforms as T
from transformers import AutoImageProcessor
from torchvision.models import resnet50, ResNet50_Weights

class ImageProcessor:
    def __init__(self, model_id='google/vit-base-patch16-224-in21k', pre_transform=None, randaug=False):
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.pre_transform = pre_transform
    
    def __call__(self, image):
        if self.pre_transform:
            image = self.pre_transform(image)
        
        image = PILImage.fromarray(image)
        image = self.processor(images=image, return_tensors='pt')['pixel_values'][0]  # [C, H, W]
        image = self.randaug(image)  # [C, H, W]
        return image