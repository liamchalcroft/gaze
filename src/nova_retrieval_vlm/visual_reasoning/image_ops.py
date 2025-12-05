from __future__ import annotations

import numpy as np
from beartype import beartype
from PIL import Image
from PIL import ImageEnhance

# Image processing constants
MIN_CROP_SIZE = 10
CROP_PADDING = 5
MAX_INTENSITY = 255
MIN_IMAGE_SIZE = 10

# Zoom factor bounds - aligned with tools.py validation
MIN_ZOOM_FACTOR = 0.5
MAX_ZOOM_FACTOR = 4.0

# Contrast factor bounds - aligned with tools.py validation
MIN_CONTRAST_FACTOR = 0.5
MAX_CONTRAST_FACTOR = 3.0


@beartype
def zoom_image(image: Image.Image, factor: float) -> Image.Image:
    """Return a zoomed version of *image* by scaling with *factor*.

    Args:
        image: Input PIL Image
        factor: Zoom factor (must be > 0)

    Returns:
        Zoomed image

    Raises:
        ValueError: If factor is not positive
    """
    if factor <= 0:
        raise ValueError("factor must be > 0")

    width, height = image.size
    new_size = (int(width * factor), int(height * factor))

    # Ensure minimum size to prevent degenerate images
    new_size = (max(MIN_IMAGE_SIZE, new_size[0]), max(MIN_IMAGE_SIZE, new_size[1]))

    return image.resize(new_size, Image.LANCZOS)


@beartype
def crop_image(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """Crop *image* using normalized coordinates (x1, y1, x2, y2) in range 0-1."""
    width, height = image.size

    # Guard against images too small to crop meaningfully
    if width < MIN_CROP_SIZE or height < MIN_CROP_SIZE:
        raise ValueError(
            f"Image too small to crop: {width}x{height}. "
            f"Minimum size is {MIN_CROP_SIZE}x{MIN_CROP_SIZE} pixels."
        )

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
    if x2 - x1 < MIN_CROP_SIZE:
        center_x = (x1 + x2) // 2
        x1 = max(0, center_x - CROP_PADDING)
        x2 = min(width, center_x + CROP_PADDING)
    if y2 - y1 < MIN_CROP_SIZE:
        center_y = (y1 + y2) // 2
        y1 = max(0, center_y - CROP_PADDING)
        y2 = min(height, center_y + CROP_PADDING)

    return image.crop((x1, y1, x2, y2))


@beartype
def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast of *image* by *factor*.

    Args:
        image: Input PIL Image
        factor: Contrast factor (must be > 0). 1.0 = no change,
                >1.0 increases contrast, <1.0 decreases contrast.

    Returns:
        Contrast-adjusted image

    Raises:
        ValueError: If factor is not positive
    """
    if factor <= 0:
        raise ValueError("factor must be > 0")

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


@beartype
def apply_intensity_threshold(image: Image.Image, lower: int, upper: int) -> Image.Image:
    """Apply intensity threshold to grayscale image and rescale to 0-255.

    Args:
        image: Input PIL Image
        lower: Lower intensity bound (0-254)
        upper: Upper intensity bound (must be > lower, max 255)

    Returns:
        Thresholded grayscale image with intensities rescaled to 0-255

    Raises:
        ValueError: If lower < 0 or upper <= lower
    """
    if lower < 0 or upper <= lower:
        raise ValueError("invalid intensity range: lower must be >= 0 and upper must be > lower")

    gray = image.convert("L")
    arr = np.array(gray)
    arr = np.clip(arr, lower, upper)
    arr = ((arr - lower) / (upper - lower) * 255).astype(np.uint8)
    return Image.fromarray(arr)


@beartype
def flip_horizontal(image: Image.Image) -> Image.Image:
    """Flip image horizontally (left-right mirror)."""
    return image.transpose(Image.FLIP_LEFT_RIGHT)


@beartype
def flip_vertical(image: Image.Image) -> Image.Image:
    """Flip image vertically (top-bottom mirror)."""
    return image.transpose(Image.FLIP_TOP_BOTTOM)


@beartype
def rotate_90(image: Image.Image, clockwise: bool = True) -> Image.Image:
    """Rotate image by 90 degrees.

    Args:
        image: Input image
        clockwise: If True, rotate clockwise; if False, rotate counter-clockwise

    Returns:
        Rotated image
    """
    if clockwise:
        return image.transpose(Image.ROTATE_270)
    return image.transpose(Image.ROTATE_90)
