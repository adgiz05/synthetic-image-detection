import albumentations as A
import cv2

class PatchAugmentation:
    def __init__(self, size=96):
        self.size = size
    
    def __call__(self, image):
        return A.Compose([
            # Normalize patch size
            A.PadIfNeeded(self.size, self.size),
            A.RandomCrop(self.size, self.size),

            # Social network downscaling (simulated with downscale + resize)
            A.OneOf([
                A.Downscale(scale_min=0.5, scale_max=0.9, p=1.0),  # hard downscale
                A.NoOp(p=1.0),
            ], p=0.5),

            # Random resize
            A.OneOf([
                A.Resize(self.size, self.size, interpolation=cv2.INTER_NEAREST),
                A.Resize(self.size, self.size, interpolation=cv2.INTER_LINEAR),
                A.Resize(self.size, self.size, interpolation=cv2.INTER_CUBIC),
                A.Resize(self.size, self.size, interpolation=cv2.INTER_AREA),
            ], p=0.5),

            # Compression + Noise (simula compresión con pérdida en redes sociales)
            A.OneOf([
                A.ImageCompression(quality_lower=40, quality_upper=95, compression_type=0, p=1),  # JPEG
                A.ImageCompression(quality_lower=40, quality_upper=95, compression_type=1, p=1),  # WebP
            ], p=0.7),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(var_limit=(2.0, 6.0), p=1.0),
                A.NoOp(p=1.0),
            ], p=0.4),

            # Geometric transformations
            A.RandomRotate90(p=0.33),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ], p=0.33),

            # A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02, p=0.2),
        ], p=1.0)(image=image)['image']