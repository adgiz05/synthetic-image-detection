import albumentations as A

def default_augmentation():
    return A.Compose([
            A.PadIfNeeded(96, 96),
            A.RandomCrop(96, 96),
            A.OneOf([
                A.OneOf([
                    A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=0, p=1),
                    A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=1, p=1),
                ], p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(3.0, 10.0), p=1.0),
            ], p=0.5),
            A.RandomRotate90(p=0.33),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ], p=0.33),
        ], p=1.0)