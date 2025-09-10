from __future__ import annotations

import numpy as np
from beartype import beartype
from PIL import Image
from PIL import ImageEnhance


@beartype
def zoom_image(image: Image.Image, factor: float) -> Image.Image:
    """Return a zoomed version of *image* by scaling with *factor*."""
    if factor <= 0:
        raise ValueError("factor must be > 0")

    # Clamp factor to reasonable bounds to prevent extreme scaling
    factor = max(0.1, min(5.0, factor))

    width, height = image.size
    new_size = (int(width * factor), int(height * factor))

    # Ensure minimum size
    new_size = (max(10, new_size[0]), max(10, new_size[1]))

    return image.resize(new_size, Image.LANCZOS)


@beartype
def crop_image(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """Crop *image* using normalized coordinates (x1, y1, x2, y2) in range 0-1."""
    width, height = image.size

    # Convert normalized coordinates to pixel coordinates
    x1 = int(box[0] * width)
    y1 = int(box[1] * height)
    x2 = int(box[2] * width)
    y2 = int(box[3] * height)

    # Ensure valid bounds
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    # Ensure minimum crop size
    if x2 - x1 < 10:
        center_x = (x1 + x2) // 2
        x1 = max(0, center_x - 5)
        x2 = min(width, center_x + 5)
    if y2 - y1 < 10:
        center_y = (y1 + y2) // 2
        y1 = max(0, center_y - 5)
        y2 = min(height, center_y + 5)

    return image.crop((x1, y1, x2, y2))


@beartype
def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast of *image* by *factor*."""
    if factor <= 0:
        raise ValueError("factor must be > 0")

    # Clamp factor to reasonable bounds
    factor = max(0.1, min(3.0, factor))

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


@beartype
def apply_intensity_threshold(image: Image.Image, lower: int, upper: int) -> Image.Image:
    """Apply intensity threshold to grayscale image and rescale to 0-255."""
    if lower < 0 or upper > 255 or lower > upper:
        raise ValueError("invalid intensity range")

    # Ensure reasonable bounds
    lower = max(0, min(254, lower))
    upper = max(lower + 1, min(255, upper))

    gray = image.convert("L")
    arr = np.array(gray)
    arr = np.clip(arr, lower, upper)
    if upper > lower:
        arr = ((arr - lower) / (upper - lower) * 255).astype(np.uint8)
    else:
        arr = np.zeros_like(arr, dtype=np.uint8)
    return Image.fromarray(arr)
