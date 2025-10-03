import cv2
import numpy as np
import albumentations as A

class PatchAugmentation:
    def __init__(self,
                 size: int = 96, # Final patch size (squared)
                 transformations: bool = True, # Whether to apply transformations
                 downscaling_prob: float = 0.3, # Probability of simulating low-res (OSNs)
                 resize_prob: float = 0.3, # Probability of random resize with different interpolations
                 compression_prob: float = 0.3, # Probability of compression artifacts
                 blur_noise_prob: float = 0.1, # Probability of blur or noise
                 color_prob: float = 0.1, # Probability of color transformations
                 texture_prob: float = 0.1, # Probability of texture transformations
                 local_artifacts_prob: float = 0.1, # Probability of local artifacts
                 rotation_prob: float = 0.1, # Probability of rotation
                 flip_prob: float = 0.1, # Probability of flip
                 ):
        self.size = size
        self.transformations = transformations
        self.downscaling_prob = downscaling_prob
        self.resize_prob = resize_prob
        self.compression_prob = compression_prob
        self.blur_noise_prob = blur_noise_prob
        self.color_prob = color_prob
        self.texture_prob = texture_prob
        self.local_artifacts_prob = local_artifacts_prob
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob

    def __call__(self, image):
        if image.dtype != np.uint8: # Force albumentations works with uint8
            image = np.clip(image, 0, 255).astype(np.uint8)

        transforms = [
            *([
                # Downscaling (simulate low-res upload)
                A.OneOf([
                    A.Downscale(scale_min=0.5, scale_max=0.9, p=1.0),
                    A.NoOp(p=1.0),
                ], p=self.downscaling_prob),

                # Random resize with different interpolations
                A.OneOf([
                    A.Resize(self.size, self.size, interpolation=cv2.INTER_NEAREST),
                    A.Resize(self.size, self.size, interpolation=cv2.INTER_LINEAR),
                    A.Resize(self.size, self.size, interpolation=cv2.INTER_CUBIC),
                    A.Resize(self.size, self.size, interpolation=cv2.INTER_AREA),
                ], p=self.resize_prob),

                # Compression artifacts
                A.OneOf([
                    A.ImageCompression(quality_lower=40, quality_upper=95, compression_type=0, p=1),  # JPEG
                    A.ImageCompression(quality_lower=40, quality_upper=95, compression_type=1, p=1),  # WebP
                ], p=self.compression_prob),

                # Blur or noise
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.GaussNoise(var_limit=(2.0, 6.0), p=1.0),
                    A.NoOp(p=1.0),
                ], p=self.blur_noise_prob),

                # Color transformations
                A.OneOf([
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=1.0),
                    A.ToGray(p=1.0),
                    A.NoOp(p=1.0),
                ], p=self.color_prob),
                
                # Texture transformations
                A.OneOf([
                    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3), p=1.0),
                    A.Emboss(alpha=(0.1, 0.3), strength=(0.5, 1.0), p=1.0),
                    A.NoOp(p=1.0),
                ], p=self.texture_prob),

                # Local artifacts
                A.OneOf([
                    A.Cutout(num_holes=1, max_h_size=32, max_w_size=32, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.ElasticTransform(alpha=20, sigma=5, alpha_affine=10, p=1.0),
                    A.NoOp(p=1.0),
                ], p=self.local_artifacts_prob),

                # Geometric transformations
                A.Rotate(limit=45, p=0.33),
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ], p=self.flip_prob)

            ] if self.transformations else []),

            # Final patch extraction
            A.PadIfNeeded(self.size, self.size),
            A.RandomCrop(self.size, self.size),
        ]

        return A.Compose(transforms, p=1.0)(image=image)["image"]
