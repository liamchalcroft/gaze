from torchvision import transforms

# ImageNet normalization parameters
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# Default transforms – keep ORIGINAL resolution (no Resize)
default_transforms: transforms.Compose = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# No separate training transforms – zero-shot setting

__all__ = [
    "default_transforms",
]
