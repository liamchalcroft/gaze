from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance


def zoom_image(image: Image.Image, factor: float) -> Image.Image:
    """Return a zoomed version of *image* by scaling with *factor*."""
    if factor <= 0:
        raise ValueError("factor must be > 0")
    width, height = image.size
    new_size = (int(width * factor), int(height * factor))
    return image.resize(new_size, Image.LANCZOS)


def crop_image(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Crop *image* using (x1, y1, x2, y2)."""
    return image.crop(box)


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast of *image* by *factor*."""
    if factor <= 0:
        raise ValueError("factor must be > 0")
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def apply_intensity_threshold(image: Image.Image, lower: int, upper: int) -> Image.Image:
    """Apply intensity threshold to grayscale image and rescale to 0-255."""
    if lower < 0 or upper > 255 or lower > upper:
        raise ValueError("invalid intensity range")
    gray = image.convert("L")
    arr = np.array(gray)
    arr = np.clip(arr, lower, upper)
    if upper > lower:
        arr = ((arr - lower) / (upper - lower) * 255).astype(np.uint8)
    else:
        arr = np.zeros_like(arr, dtype=np.uint8)
    return Image.fromarray(arr)
