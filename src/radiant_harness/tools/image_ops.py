"""Image manipulation operations for visual tools.

Provides core image operations: zoom, crop, contrast, threshold, flip, rotate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from beartype import beartype
from PIL import Image
from PIL import ImageEnhance

from radiant_harness.config import ImageProcessingConfig
from radiant_harness.config import get_config

if TYPE_CHECKING:
    pass


@beartype
def _get_image_config() -> ImageProcessingConfig:
    """Get the current image processing configuration."""
    return get_config().image


@beartype
def zoom_image(
    image: Image.Image,
    factor: float,
    config: ImageProcessingConfig | None = None,
) -> Image.Image:
    """Return a zoomed version of *image* by scaling with *factor*.

    Args:
        image: Input PIL Image
        factor: Zoom factor (must be in configured range)
        config: Optional image config. If None, uses global default.

    Returns:
        Zoomed image

    Raises:
        ValueError: If factor is out of valid range

    Example:
        from PIL import Image
        from radiant_harness.tools.image_ops import zoom_image

        img = Image.open("scan.png")
        zoomed = zoom_image(img, 2.0)  # 2x magnification
    """
    cfg = config or _get_image_config()

    if not cfg.min_zoom_factor <= factor <= cfg.max_zoom_factor:
        raise ValueError(
            f"factor must be in range [{cfg.min_zoom_factor}, {cfg.max_zoom_factor}], got {factor}"
        )

    width, height = image.size
    new_size = (int(width * factor), int(height * factor))

    # Ensure minimum size to prevent degenerate images
    new_size = (max(cfg.min_image_size, new_size[0]), max(cfg.min_image_size, new_size[1]))

    return image.resize(new_size, Image.LANCZOS)


@beartype
def crop_image(
    image: Image.Image,
    box: tuple[float, float, float, float],
    config: ImageProcessingConfig | None = None,
) -> Image.Image:
    """Crop *image* using normalized coordinates (x1, y1, x2, y2) in range 0-1.

    Args:
        image: Input PIL Image
        box: Tuple of (x1, y1, x2, y2) normalized coordinates in range [0, 1]
             where x2 > x1 and y2 > y1
        config: Optional image config. If None, uses global default.

    Returns:
        Cropped image

    Raises:
        ValueError: If image is too small to crop, coordinates are out of range,
                   or resulting crop would be too small

    Example:
        from PIL import Image
        from radiant_harness.tools.image_ops import crop_image

        img = Image.open("scan.png")
        # Crop center 50% of the image
        cropped = crop_image(img, (0.25, 0.25, 0.75, 0.75))
    """
    cfg = config or _get_image_config()
    min_size = cfg.min_image_size

    width, height = image.size
    x1_norm, y1_norm, x2_norm, y2_norm = box

    # Validate normalized coordinates are in valid range
    if not all(0 <= coord <= 1 for coord in box):
        raise ValueError(f"All coordinates must be in range [0, 1], got box={box}")

    # Validate ordering
    if x2_norm <= x1_norm or y2_norm <= y1_norm:
        raise ValueError(f"Invalid crop box: x2 must be > x1 and y2 must be > y1, got box={box}")

    # Guard against images too small to crop meaningfully
    if width < min_size or height < min_size:
        raise ValueError(
            f"Image too small to crop: {width}x{height}. "
            f"Minimum size is {min_size}x{min_size} pixels."
        )

    # Convert normalized coordinates to pixel coordinates
    x1 = int(x1_norm * width)
    y1 = int(y1_norm * height)
    x2 = int(x2_norm * width)
    y2 = int(y2_norm * height)

    # Clamp to image bounds (handles floating point edge cases)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    # Validate resulting crop size
    crop_width = x2 - x1
    crop_height = y2 - y1
    if crop_width < min_size or crop_height < min_size:
        raise ValueError(
            f"Resulting crop region too small: {crop_width}x{crop_height} pixels. "
            f"Minimum size is {min_size}x{min_size} pixels. "
            f"Consider using a larger crop region."
        )

    return image.crop((x1, y1, x2, y2))


@beartype
def adjust_contrast(
    image: Image.Image,
    factor: float,
    config: ImageProcessingConfig | None = None,
) -> Image.Image:
    """Adjust contrast of *image* by *factor*.

    Args:
        image: Input PIL Image
        factor: Contrast factor (must be in configured range).
                1.0 = no change, >1.0 increases contrast, <1.0 decreases contrast.
        config: Optional image config. If None, uses global default.

    Returns:
        Contrast-adjusted image

    Raises:
        ValueError: If factor is out of valid range

    Example:
        from PIL import Image
        from radiant_harness.tools.image_ops import adjust_contrast

        img = Image.open("scan.png")
        enhanced = adjust_contrast(img, 1.5)  # 50% more contrast
    """
    cfg = config or _get_image_config()

    if not cfg.min_contrast_factor <= factor <= cfg.max_contrast_factor:
        raise ValueError(
            f"factor must be in range [{cfg.min_contrast_factor}, {cfg.max_contrast_factor}], "
            f"got {factor}"
        )

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

    Example:
        from PIL import Image
        from radiant_harness.tools.image_ops import apply_intensity_threshold

        img = Image.open("scan.png")
        # Window to intensities 50-200
        windowed = apply_intensity_threshold(img, 50, 200)
    """
    if lower < 0 or upper <= lower:
        raise ValueError("invalid intensity range: lower must be >= 0 and upper must be > lower")

    gray = image.convert("L")
    arr = np.array(gray)
    arr = np.clip(arr, lower, upper)
    # Rescale to 0-255 with explicit clipping to prevent floating point overflow
    arr = np.clip((arr - lower) / (upper - lower) * 255, 0, 255).astype(np.uint8)
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
