"""Visual tools for radiology image analysis.

Provides zoom, crop, contrast adjustment, thresholding, flip, and rotate tools
for interactive image analysis by VLMs.

This module combines:
- Core image operations (zoom, crop, contrast, threshold, flip, rotate)
- Tool wrappers for VLM integration
- Tool factory (create_visual_tools)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from beartype import beartype
from loguru import logger
from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageFont
from PIL import ImageOps

from radiant_harness.config import ImageProcessingConfig
from radiant_harness.config import get_config
from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.tools.image_manager import ImageManager
from radiant_harness.tools.registry import EncodedImage
from radiant_harness.tools.registry import encode_image
from radiant_harness.tools.tool import Tool
from radiant_harness.types import ToolResult

if TYPE_CHECKING:
    from radiant_harness.tools.registry import ToolRegistry


# =============================================================================
# Image Operations (core functions)
# =============================================================================


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
    """
    cfg = config or _get_image_config()

    if not cfg.min_zoom_factor <= factor <= cfg.max_zoom_factor:
        raise ValueError(
            f"factor must be in range [{cfg.min_zoom_factor}, {cfg.max_zoom_factor}], got {factor}"
        )

    width, height = image.size
    new_w = int(width * factor)
    new_h = int(height * factor)

    # Reject zooms that would exceed the maximum allowed dimension.
    max_dim = cfg.max_image_dimension
    if new_w > max_dim or new_h > max_dim:
        raise ValueError(
            f"Zoom would produce {new_w}x{new_h} which exceeds "
            f"max_image_dimension={max_dim}. Use a smaller factor or crop first."
        )

    # Ensure minimum size while preserving aspect ratio.
    # Per-axis clamping would distort the image, which is unacceptable
    # for diagnostic medical imaging where geometry must be faithful.
    min_sz = cfg.min_image_size
    if new_w < min_sz or new_h < min_sz:
        safe_w = max(1, new_w)
        safe_h = max(1, new_h)
        scale_up = max(min_sz / safe_w, min_sz / safe_h)
        new_w = max(min_sz, int(safe_w * scale_up))
        new_h = max(min_sz, int(safe_h * scale_up))

    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


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
    """
    cfg = config or _get_image_config()
    min_size = cfg.min_image_size

    width, height = image.size
    x1_norm, y1_norm, x2_norm, y2_norm = box

    if not all(0 <= coord <= 1 for coord in box):
        raise ValueError(f"All coordinates must be in range [0, 1], got box={box}")

    if x2_norm <= x1_norm or y2_norm <= y1_norm:
        raise ValueError(f"Invalid crop box: x2 must be > x1 and y2 must be > y1, got box={box}")

    if width < min_size or height < min_size:
        raise ValueError(
            f"Image too small to crop: {width}x{height}. "
            f"Minimum size is {min_size}x{min_size} pixels."
        )

    x1 = int(x1_norm * width)
    y1 = int(y1_norm * height)
    x2 = int(x2_norm * width)
    y2 = int(y2_norm * height)

    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    crop_width = x2 - x1
    crop_height = y2 - y1
    if crop_width < min_size or crop_height < min_size:
        raise ValueError(
            f"Resulting crop region too small: {crop_width}x{crop_height} pixels. "
            f"Minimum size is {min_size}x{min_size} pixels."
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
def apply_intensity_threshold(
    image: Image.Image,
    lower: int,
    upper: int,
    config: ImageProcessingConfig | None = None,
) -> Image.Image:
    """Apply intensity threshold to grayscale image and rescale to 0-255.

    Args:
        image: Input PIL Image
        lower: Lower intensity bound (0-254)
        upper: Upper intensity bound (must be > lower, max 255)
        config: Optional config for min_threshold_window. Uses global default if None.

    Returns:
        Thresholded grayscale image with intensities rescaled to 0-255

    Raises:
        ValueError: If bounds are invalid or window width is below minimum
    """
    if lower < 0:
        raise ValueError(f"lower must be >= 0, got {lower}")
    if upper > 255:
        raise ValueError(f"upper must be <= 255, got {upper}")
    if upper <= lower:
        raise ValueError(f"upper must be > lower, got lower={lower}, upper={upper}")

    cfg = config or get_config().image
    window_width = upper - lower
    if window_width < cfg.min_threshold_window:
        raise ValueError(
            f"Threshold window width {window_width} (upper={upper} - lower={lower}) "
            f"is below minimum {cfg.min_threshold_window}. "
            f"Narrow windows destroy diagnostic information."
        )

    gray = image.convert("L")
    arr = np.array(gray)
    arr = np.clip(arr, lower, upper)
    # Rescale to 0-255 with explicit clipping to prevent floating point overflow
    arr = np.clip((arr - lower) / (upper - lower) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


@beartype
def flip_horizontal(image: Image.Image) -> Image.Image:
    """Flip image horizontally (left-right mirror)."""
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


@beartype
def flip_vertical(image: Image.Image) -> Image.Image:
    """Flip image vertically (top-bottom mirror)."""
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


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
        return image.transpose(Image.Transpose.ROTATE_270)
    return image.transpose(Image.Transpose.ROTATE_90)


@beartype
def adjust_brightness(
    image: Image.Image,
    factor: float,
    config: ImageProcessingConfig | None = None,
) -> Image.Image:
    """Adjust brightness of *image* by *factor*.

    Args:
        image: Input PIL Image
        factor: Brightness factor (must be in configured range).
                1.0 = no change, >1.0 increases brightness, <1.0 decreases.
        config: Optional image config. If None, uses global default.

    Returns:
        Brightness-adjusted image

    Raises:
        ValueError: If factor is out of valid range
    """
    cfg = config or _get_image_config()

    if not cfg.min_brightness_factor <= factor <= cfg.max_brightness_factor:
        raise ValueError(
            f"factor must be in range [{cfg.min_brightness_factor}, {cfg.max_brightness_factor}], "
            f"got {factor}"
        )

    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


@beartype
def adjust_sharpness(
    image: Image.Image,
    factor: float,
    config: ImageProcessingConfig | None = None,
) -> Image.Image:
    """Adjust sharpness of *image* by *factor*.

    Args:
        image: Input PIL Image
        factor: Sharpness factor (must be in configured range).
                0.0 = blurred, 1.0 = original, >1.0 = sharpened.
        config: Optional image config. If None, uses global default.

    Returns:
        Sharpness-adjusted image

    Raises:
        ValueError: If factor is out of valid range
    """
    cfg = config or _get_image_config()

    if not cfg.min_sharpness_factor <= factor <= cfg.max_sharpness_factor:
        raise ValueError(
            f"factor must be in range [{cfg.min_sharpness_factor}, {cfg.max_sharpness_factor}], "
            f"got {factor}"
        )

    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


@beartype
def equalize_histogram(image: Image.Image) -> Image.Image:
    """Equalize the histogram of *image* for improved contrast distribution.

    Converts to grayscale first since brain MRI is inherently grayscale;
    per-channel equalization on RGB is meaningless for medical imaging.

    Args:
        image: Input PIL Image

    Returns:
        Histogram-equalized grayscale image
    """
    gray = image.convert("L")
    return ImageOps.equalize(gray)


@beartype
def get_intensity_stats(
    image: Image.Image,
    box: tuple[float, float, float, float] | None = None,
) -> dict[str, object]:
    """Compute intensity statistics over *image* or a sub-region.

    Args:
        image: Input PIL Image
        box: Optional normalized coordinates (x1, y1, x2, y2) in [0, 1] for a sub-region.
             If None, computes stats over the full image.

    Returns:
        Dict with mean, std, min, max, median, and 10-bin histogram.

    Raises:
        ValueError: If box coordinates are invalid
    """
    gray = np.array(image.convert("L"))

    if box is not None:
        x1_n, y1_n, x2_n, y2_n = box
        if not all(0 <= c <= 1 for c in box):
            raise ValueError(f"All box coordinates must be in [0, 1], got {box}")
        if x2_n <= x1_n or y2_n <= y1_n:
            raise ValueError(f"Invalid box: x2 must be > x1 and y2 must be > y1, got {box}")
        h, w = gray.shape
        x1 = int(x1_n * w)
        y1 = int(y1_n * h)
        x2 = max(x1 + 1, int(x2_n * w))
        y2 = max(y1 + 1, int(y2_n * h))
        gray = gray[y1:y2, x1:x2]

    histogram, _ = np.histogram(gray, bins=10, range=(0, 255))
    return {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "min": int(np.min(gray)),
        "max": int(np.max(gray)),
        "median": float(np.median(gray)),
        "histogram": histogram.tolist(),
    }


@beartype
def measure_distance(
    image: Image.Image,
    point1: tuple[float, float],
    point2: tuple[float, float],
) -> dict[str, object]:
    """Measure Euclidean distance between two points on *image*.

    Args:
        image: Input PIL Image (used for dimension scaling)
        point1: Normalized (x, y) coordinates in [0, 1]
        point2: Normalized (x, y) coordinates in [0, 1]

    Returns:
        Dict with distance_pixels, point1_pixels, point2_pixels, image_size.

    Raises:
        ValueError: If coordinates are out of [0, 1] range
    """
    for name, pt in [("point1", point1), ("point2", point2)]:
        if not (0 <= pt[0] <= 1 and 0 <= pt[1] <= 1):
            raise ValueError(f"{name} coordinates must be in [0, 1], got {pt}")

    w, h = image.size
    p1_px = (point1[0] * w, point1[1] * h)
    p2_px = (point2[0] * w, point2[1] * h)
    dist = ((p2_px[0] - p1_px[0]) ** 2 + (p2_px[1] - p1_px[1]) ** 2) ** 0.5

    return {
        "distance_pixels": float(dist),
        "point1_pixels": p1_px,
        "point2_pixels": p2_px,
        "image_size": (w, h),
    }


@beartype
def draw_grid_overlay(
    image: Image.Image,
    divisions: int,
    config: ImageProcessingConfig | None = None,
) -> Image.Image:
    """Draw a labeled grid overlay on *image*.

    Draws a divisions x divisions grid with cell labels (A1, B2, etc.).
    Green lines and yellow text on black background for readability on MRI.
    Works on a copy; does not mutate the input.

    Args:
        image: Input PIL Image
        divisions: Number of grid divisions per axis (rows and columns)
        config: Optional image config for max_grid_divisions. If None, uses global default.

    Returns:
        Image with grid overlay (RGB mode)

    Raises:
        ValueError: If divisions is out of valid range
    """
    cfg = config or _get_image_config()

    if divisions < 2:
        raise ValueError(f"divisions must be >= 2, got {divisions}")
    if divisions > cfg.max_grid_divisions:
        raise ValueError(f"divisions must be <= {cfg.max_grid_divisions}, got {divisions}")

    # Work on RGB copy
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    w, h = result.size
    font = ImageFont.load_default()

    # Draw grid lines
    grid_color = (0, 255, 0)  # green
    for i in range(1, divisions):
        x = int(i * w / divisions)
        draw.line([(x, 0), (x, h)], fill=grid_color, width=1)
        y = int(i * h / divisions)
        draw.line([(0, y), (w, y)], fill=grid_color, width=1)

    # Label cells
    label_color = (255, 255, 0)  # yellow
    bg_color = (0, 0, 0)  # black
    for row in range(divisions):
        for col in range(divisions):
            label = f"{chr(65 + col)}{row + 1}"
            cx = int((col + 0.5) * w / divisions)
            cy = int((row + 0.5) * h / divisions)
            bbox = font.getbbox(label)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = cx - tw // 2
            ty = cy - th // 2
            draw.rectangle([tx - 1, ty - 1, tx + tw + 1, ty + th + 1], fill=bg_color)
            draw.text((tx, ty), label, fill=label_color, font=font)

    return result


@beartype
def detect_edges(
    image: Image.Image,
    method: str = "sobel",
) -> Image.Image:
    """Detect edges in *image* using Sobel or Laplacian operators.

    Args:
        image: Input PIL Image
        method: Edge detection method ("sobel" or "laplacian")

    Returns:
        Grayscale edge map image

    Raises:
        ValueError: If method is not recognized
    """
    if method not in ("sobel", "laplacian"):
        raise ValueError(f"method must be 'sobel' or 'laplacian', got {method!r}")

    gray = np.array(image.convert("L"), dtype=np.float64)

    if method == "sobel":
        # Sobel kernels
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        # Pad to handle borders
        padded = np.pad(gray, 1, mode="edge")
        h, w = gray.shape
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        for i in range(3):
            for j in range(3):
                gx += kx[i, j] * padded[i : i + h, j : j + w]
                gy += ky[i, j] * padded[i : i + h, j : j + w]
        magnitude = np.sqrt(gx**2 + gy**2)
    else:  # laplacian
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        padded = np.pad(gray, 1, mode="edge")
        h, w = gray.shape
        result = np.zeros_like(gray)
        for i in range(3):
            for j in range(3):
                result += kernel[i, j] * padded[i : i + h, j : j + w]
        magnitude = np.abs(result)

    # Normalize to 0-255
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255
    return Image.fromarray(magnitude.astype(np.uint8))


@beartype
def compute_symmetry_diff(image: Image.Image) -> Image.Image:
    """Compute left-right symmetry difference map.

    Flips the image horizontally and computes the absolute pixel-wise
    difference, highlighting asymmetric regions (potential pathology).

    Args:
        image: Input PIL Image

    Returns:
        Grayscale difference map (bright = asymmetric regions)
    """
    gray = np.array(image.convert("L"), dtype=np.float64)
    flipped = np.fliplr(gray)
    diff = np.abs(gray - flipped)
    # Normalize to 0-255
    if diff.max() > 0:
        diff = diff / diff.max() * 255
    return Image.fromarray(diff.astype(np.uint8))


@beartype
def annotate_region(
    image: Image.Image,
    box: tuple[float, float, float, float],
    color: str = "red",
    label: str | None = None,
) -> Image.Image:
    """Draw a bounding box annotation on *image*.

    Works on a copy; does not mutate the input.

    Args:
        image: Input PIL Image
        box: Normalized coordinates (x1, y1, x2, y2) in [0, 1]
        color: Box color name (e.g. "red", "green", "yellow")
        label: Optional text label drawn above the box

    Returns:
        Annotated image (RGB)

    Raises:
        ValueError: If box coordinates are invalid
    """
    if not all(0 <= c <= 1 for c in box):
        raise ValueError(f"All box coordinates must be in [0, 1], got {box}")
    x1_n, y1_n, x2_n, y2_n = box
    if x2_n <= x1_n or y2_n <= y1_n:
        raise ValueError(f"Invalid box: x2 must be > x1 and y2 must be > y1, got {box}")

    result = image.convert("RGB").copy()
    w, h = result.size
    draw = ImageDraw.Draw(result)

    x1 = int(x1_n * w)
    y1 = int(y1_n * h)
    x2 = int(x2_n * w)
    y2 = int(y2_n * h)

    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    if label:
        font = ImageFont.load_default()
        bbox = font.getbbox(label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        # Draw label background above box
        label_y = max(0, y1 - th - 4)
        draw.rectangle([x1, label_y, x1 + tw + 4, label_y + th + 4], fill=color)
        draw.text((x1 + 2, label_y + 2), label, fill="white", font=font)

    return result


@beartype
def invert_image(image: Image.Image) -> Image.Image:
    """Invert pixel intensities (negative image).

    Useful for toggling between standard and inverted display modes
    common in radiology viewers.

    Args:
        image: Input PIL Image

    Returns:
        Inverted grayscale image
    """
    return ImageOps.invert(image.convert("L"))


# Clinical window presets: (center, width)
#
# CT presets (Hounsfield units) — designed for CT data.  When applied to
# 8-bit MRI pixel values these will compress or clip the dynamic range.
# The effective_levels check in apply_window_level() catches the worst
# cases, but prefer MRI-specific presets for MRI data.
#
# MRI presets — designed for 8-bit MRI pixel values (0-255) as produced
# by standard DICOM-to-PNG/JPEG conversion pipelines.
WINDOW_PRESETS: dict[str, tuple[int, int]] = {
    # CT presets
    "brain": (40, 80),
    "subdural": (75, 215),
    "bone": (400, 1800),
    "soft_tissue": (40, 400),
    "stroke": (32, 40),
    "posterior_fossa": (36, 120),
    # MRI presets (8-bit pixel values)
    "mri_brain": (128, 230),
    "mri_flair": (110, 200),
    "mri_t2": (140, 220),
}


@beartype
def apply_window_level(
    image: Image.Image,
    center: int | None = None,
    width: int | None = None,
    preset: str | None = None,
) -> Image.Image:
    """Apply clinical window/level to *image*.

    Either provide center+width or a preset name. Preset names:
    brain, subdural, bone, soft_tissue, stroke, posterior_fossa.

    Args:
        image: Input PIL Image
        center: Window center intensity
        width: Window width
        preset: Clinical preset name (overrides center/width)

    Returns:
        Windowed grayscale image

    Raises:
        ValueError: If neither preset nor center+width provided, or invalid preset
    """
    if preset is not None:
        if preset not in WINDOW_PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}. Available: {sorted(WINDOW_PRESETS.keys())}"
            )
        if center is not None or width is not None:
            logger.warning(
                "window_level: preset={!r} overrides center={}/width={}",
                preset,
                center,
                width,
            )
        center, width = WINDOW_PRESETS[preset]
    elif center is None or width is None:
        raise ValueError("Must provide either preset or both center and width")

    # Safety floor: ALL window widths (including presets) must meet the
    # minimum.  Presets are curated to comply; if one is below the floor
    # it indicates a configuration error, not an intentional override.
    cfg = _get_image_config()
    if width < cfg.min_window_width:
        raise ValueError(
            f"width must be >= {cfg.min_window_width}, got {width}. "
            f"Very narrow windows destroy diagnostic information."
        )

    lower = center - width / 2
    upper = center + width / 2

    gray = np.array(image.convert("L"), dtype=np.float64)

    # Check that the window meaningfully covers the image's actual data range.
    # CT presets (e.g. bone: center=400, width=1800) applied to 8-bit images
    # compress the output to very few levels, producing misleading results.
    # Skip for uniform images (img_min == img_max): the result is deterministic
    # regardless of window settings and there is no information to destroy.
    img_min, img_max = float(gray.min()), float(gray.max())
    if img_min < img_max:
        effective_lower = max(lower, img_min)
        effective_upper = min(upper, img_max)
        if effective_upper <= effective_lower:
            raise ValueError(
                f"Window [center={center}, width={width}] does not overlap with "
                f"image intensity range [{img_min:.0f}, {img_max:.0f}]. "
                f"No data would be visible."
            )
        effective_levels = int((effective_upper - effective_lower) / (upper - lower) * 255)
        if effective_levels < cfg.min_window_width:
            raise ValueError(
                f"Window [center={center}, width={width}] compresses image range "
                f"[{img_min:.0f}, {img_max:.0f}] to only {effective_levels} output "
                f"levels (minimum: {cfg.min_window_width}). Use a narrower window "
                f"suited to this image's bit depth."
            )

    gray = np.clip(gray, lower, upper)
    gray = (gray - lower) / (upper - lower) * 255 if upper > lower else np.zeros_like(gray)
    return Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))


@beartype
def adaptive_equalize(
    image: Image.Image,
    clip_limit: float = 2.0,
    tile_size: int = 8,
    config: ImageProcessingConfig | None = None,
) -> Image.Image:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Operates on local tiles for better local contrast than global equalization.
    Particularly useful for brain MRI white matter lesions.

    Args:
        image: Input PIL Image
        clip_limit: Histogram clip limit (higher = more contrast)
        tile_size: Tile grid size (image divided into tile_size x tile_size tiles)
        config: Optional config for clip_limit bounds

    Returns:
        CLAHE-processed grayscale image

    Raises:
        ValueError: If clip_limit or tile_size is out of range
    """
    cfg = config or _get_image_config()

    if not cfg.min_clahe_clip_limit <= clip_limit <= cfg.max_clahe_clip_limit:
        raise ValueError(
            f"clip_limit must be in [{cfg.min_clahe_clip_limit}, {cfg.max_clahe_clip_limit}], "
            f"got {clip_limit}"
        )
    if tile_size < 2 or tile_size > cfg.max_clahe_tile_size:
        raise ValueError(f"tile_size must be in [2, {cfg.max_clahe_tile_size}], got {tile_size}")

    gray = np.array(image.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # Compute tile dimensions
    th = max(1, h // tile_size)
    tw = max(1, w // tile_size)
    n_bins = 256

    # Build per-tile CDFs
    cdfs = np.zeros((tile_size, tile_size, n_bins))
    for ty in range(tile_size):
        for tx in range(tile_size):
            y0 = ty * th
            x0 = tx * tw
            y1 = h if ty == tile_size - 1 else (ty + 1) * th
            x1 = w if tx == tile_size - 1 else (tx + 1) * tw
            tile = gray[y0:y1, x0:x1].astype(np.uint8)
            hist, _ = np.histogram(tile, bins=n_bins, range=(0, 255))

            # Clip histogram
            n_pixels = tile.size
            clip_count = max(1, int(clip_limit * n_pixels / n_bins))
            excess = np.sum(np.maximum(hist - clip_count, 0))
            hist = np.minimum(hist, clip_count)
            hist += excess // n_bins  # redistribute excess uniformly

            # Compute CDF
            cdf = hist.cumsum()
            if cdf[-1] > 0:
                cdf = cdf / cdf[-1] * 255
            cdfs[ty, tx] = cdf

    # Map each pixel using bilinear interpolation of tile CDFs (vectorized)
    py_coords = np.arange(h, dtype=np.float64)
    px_coords = np.arange(w, dtype=np.float64)
    # Shape: (h, w) via broadcasting
    fy = (py_coords[:, np.newaxis] / th) - 0.5
    fx = (px_coords[np.newaxis, :] / tw) - 0.5

    ty0 = np.clip(np.floor(fy).astype(np.intp), 0, tile_size - 1)
    ty1 = np.clip(ty0 + 1, 0, tile_size - 1)
    tx0 = np.clip(np.floor(fx).astype(np.intp), 0, tile_size - 1)
    tx1 = np.clip(tx0 + 1, 0, tile_size - 1)

    # Interpolation weights — zero when clamped to same tile index
    wy = np.where(ty0 != ty1, fy - np.floor(fy), 0.0)
    wx = np.where(tx0 != tx1, fx - np.floor(fx), 0.0)

    val = gray.astype(np.intp)

    # Gather CDF lookups for all four neighbours: cdfs[tile_y, tile_x, pixel_val]
    v00 = cdfs[ty0, tx0, val]
    v01 = cdfs[ty0, tx1, val]
    v10 = cdfs[ty1, tx0, val]
    v11 = cdfs[ty1, tx1, val]

    top = v00 * (1 - wx) + v01 * wx
    bot = v10 * (1 - wx) + v11 * wx
    result = top * (1 - wy) + bot * wy

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


@beartype
def compute_intensity_profile(
    image: Image.Image,
    point1: tuple[float, float],
    point2: tuple[float, float],
) -> dict[str, object]:
    """Sample pixel intensities along a line between two points.

    Uses Bresenham-style sampling for accurate pixel traversal.

    Args:
        image: Input PIL Image
        point1: Start point as normalized (x, y) in [0, 1]
        point2: End point as normalized (x, y) in [0, 1]

    Returns:
        Dict with profile (list of intensities), positions, stats, and pixel coords.

    Raises:
        ValueError: If coordinates are out of [0, 1]
    """
    for name, pt in [("point1", point1), ("point2", point2)]:
        if not (0 <= pt[0] <= 1 and 0 <= pt[1] <= 1):
            raise ValueError(f"{name} coordinates must be in [0, 1], got {pt}")

    gray = np.array(image.convert("L"))
    h, w = gray.shape

    x0 = int(point1[0] * (w - 1))
    y0 = int(point1[1] * (h - 1))
    x1 = int(point2[0] * (w - 1))
    y1 = int(point2[1] * (h - 1))

    # Sample along the line using linear interpolation
    n_samples = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
    xs = np.linspace(x0, x1, n_samples).astype(int)
    ys = np.linspace(y0, y1, n_samples).astype(int)
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    # Vectorized fancy-index lookup — avoids per-pixel Python overhead.
    intensity_arr = gray[ys, xs]
    intensities = intensity_arr.tolist()

    return {
        "profile": intensities,
        "n_samples": n_samples,
        "mean": float(np.mean(intensity_arr)),
        "std": float(np.std(intensity_arr)),
        "min": int(np.min(intensity_arr)),
        "max": int(np.max(intensity_arr)),
        "point1_pixels": (x0, y0),
        "point2_pixels": (x1, y1),
    }


@beartype
def denoise_gaussian(
    image: Image.Image,
    sigma: float,
    config: ImageProcessingConfig | None = None,
) -> Image.Image:
    """Apply Gaussian blur for noise reduction.

    Args:
        image: Input PIL Image
        sigma: Gaussian kernel standard deviation
        config: Optional config for sigma bounds

    Returns:
        Denoised image

    Raises:
        ValueError: If sigma is out of range
    """
    cfg = config or _get_image_config()

    if not cfg.min_gaussian_sigma <= sigma <= cfg.max_gaussian_sigma:
        raise ValueError(
            f"sigma must be in [{cfg.min_gaussian_sigma}, {cfg.max_gaussian_sigma}], got {sigma}"
        )

    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


@beartype
def morphological_op(
    image: Image.Image,
    operation: str,
    iterations: int = 1,
    threshold_value: int | None = None,
    config: ImageProcessingConfig | None = None,
) -> Image.Image:
    """Apply morphological operation (erode, dilate, open, close).

    Uses PIL's MinFilter (erosion) and MaxFilter (dilation). If threshold_value
    is provided, first binarizes the image at that intensity.

    Args:
        image: Input PIL Image
        operation: One of "erode", "dilate", "open", "close"
        iterations: Number of times to apply the operation
        threshold_value: Optional intensity threshold for binarization (0-255)
        config: Optional config for max iterations

    Returns:
        Processed grayscale image

    Raises:
        ValueError: If operation is invalid or iterations out of range
    """
    cfg = config or _get_image_config()

    valid_ops = ("erode", "dilate", "open", "close")
    if operation not in valid_ops:
        raise ValueError(f"operation must be one of {valid_ops}, got {operation!r}")
    if iterations < 1 or iterations > cfg.max_morphological_iterations:
        raise ValueError(
            f"iterations must be in [1, {cfg.max_morphological_iterations}], got {iterations}"
        )

    result = image.convert("L")

    # Optional binarization
    if threshold_value is not None:
        if not 0 <= threshold_value <= 255:
            raise ValueError(f"threshold_value must be in [0, 255], got {threshold_value}")
        arr = np.array(result)
        arr = ((arr >= threshold_value) * 255).astype(np.uint8)
        result = Image.fromarray(arr)

    def _erode(img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.MinFilter(3))

    def _dilate(img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.MaxFilter(3))

    ops: dict[str, list[Callable[[Image.Image], Image.Image]]] = {
        "erode": [_erode],
        "dilate": [_dilate],
        "open": [_erode, _dilate],  # erosion then dilation
        "close": [_dilate, _erode],  # dilation then erosion
    }

    for _ in range(iterations):
        for op_fn in ops[operation]:
            result = op_fn(result)

    return result


# Threshold for distinguishing rounding-error coords (e.g. 1.001) from pixel
# coords (e.g. 200).  Values above 1.0 but below this are clamped to [0, 1].
# The smallest valid image is 10x10 (min_image_size), so any pixel coordinate
# >= 2 could plausibly be a pixel value.  Values in (1.0, 2.0) are almost
# certainly normalized coords with small rounding overshoot.
_PIXEL_COORD_THRESHOLD = 2.0

# =============================================================================
# Tool Executors
# =============================================================================


def _require_image(registry: ToolRegistry) -> Image.Image:
    """Get current image from registry, raising if none is loaded."""
    image_manager = registry.get_image_manager()
    if image_manager.current_image is None:
        raise ToolExecutionError("Tool requires a loaded image but none is active")
    return image_manager.current_image


def _maybe_normalize_box(box: list[float], image: Image.Image) -> list[float]:
    """Auto-normalize pixel coordinates to [0, 1] if values indicate pixel space.

    Models sometimes pass pixel coordinates (e.g. [200, 150, 380, 350] or
    [0, 50, 200, 300]) instead of normalized [0, 1] values.

    Detection heuristic: values are categorised into three bands:

    1. All values in [0, 1] — already normalized, pass through.
    2. Any value > 1.0 but max(box) < ``_PIXEL_COORD_THRESHOLD`` — likely
       normalized coordinates with small rounding errors.  Clamp to [0, 1].
    3. max(box) >= ``_PIXEL_COORD_THRESHOLD`` — clearly pixel coordinates.
       Normalize by dividing by image dimensions.
    """
    max_val = max(box)
    if max_val <= 1.0:
        return box

    if max_val < _PIXEL_COORD_THRESHOLD:
        # Small overshoot (e.g. 1.001) — clamp, don't normalize.
        clamped = [max(0.0, min(v, 1.0)) for v in box]
        logger.warning(
            "Clamped near-normalized coords {} -> {} (max {:.4f} < threshold {})",
            box,
            clamped,
            max_val,
            _PIXEL_COORD_THRESHOLD,
        )
        return clamped

    w, h = image.size
    normalized = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
    logger.warning(
        "Auto-normalized pixel coords {} -> {:.3f},{:.3f},{:.3f},{:.3f} (image {}x{})",
        box,
        *normalized,
        w,
        h,
    )
    return normalized


def _maybe_normalize_point(point: list[float], image: Image.Image) -> list[float]:
    """Auto-normalize pixel coordinates to [0, 1] if values indicate pixel space.

    Same three-band heuristic as :func:`_maybe_normalize_box` but for
    2-element point arrays.
    """
    max_val = max(point)
    if max_val <= 1.0:
        return point

    if max_val < _PIXEL_COORD_THRESHOLD:
        clamped = [max(0.0, min(v, 1.0)) for v in point]
        logger.warning(
            "Clamped near-normalized point {} -> {} (max {:.4f} < threshold {})",
            point,
            clamped,
            max_val,
            _PIXEL_COORD_THRESHOLD,
        )
        return clamped

    w, h = image.size
    normalized = [point[0] / w, point[1] / h]
    logger.warning(
        "Auto-normalized pixel point {} -> {:.3f},{:.3f} (image {}x{})",
        point,
        *normalized,
        w,
        h,
    )
    return normalized


def _get_current_image(registry: ToolRegistry) -> Image.Image:
    """Get current image after a transform (guaranteed non-None by transform_image)."""
    image_manager = registry.get_image_manager()
    img = image_manager.current_image
    if img is None:
        raise ToolExecutionError("Image unexpectedly None after transform")
    return img


async def _transform_and_encode(
    registry: ToolRegistry,
    operation: Callable[[Image.Image], Image.Image],
) -> tuple[Image.Image, EncodedImage]:
    """Apply *operation* to the registry's image and encode the result.

    Both the PIL transform and the JPEG encoding are offloaded to a worker
    thread so the async event loop is never blocked by CPU-intensive work.
    """
    image_manager = registry.get_image_manager()

    def _work(mgr: ImageManager) -> tuple[Image.Image, EncodedImage]:
        mgr.transform_image(operation)
        current = mgr.current_image
        if current is None:
            raise ToolExecutionError("Image unexpectedly None after transform")
        encoded = encode_image(current)
        return current, encoded

    return await asyncio.to_thread(_work, image_manager)


async def _read_only_encode(
    registry: ToolRegistry,
    operation: Callable[[Image.Image], Image.Image],
) -> tuple[Image.Image, EncodedImage]:
    """Apply *operation* and encode the result WITHOUT mutating image state.

    Used for visualization-only tools (annotations, grid overlays) that
    should return a rendered image for the model to see without
    contaminating the image manager's state for subsequent tools.
    """
    image = _require_image(registry)

    def _work() -> tuple[Image.Image, EncodedImage]:
        result = operation(image)
        encoded = encode_image(result)
        return result, encoded

    return await asyncio.to_thread(_work)


async def _execute_zoom(registry: ToolRegistry, factor: float) -> ToolResult:
    """Execute zoom tool."""
    image = _require_image(registry)
    original_size = image.size

    try:
        current, encoded = await _transform_and_encode(
            registry, lambda img: zoom_image(img, factor)
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid zoom factor: {e}") from e

    new_size = current.size
    return ToolResult(
        tool_name="zoom",
        description=f"Zoomed {factor:.1f}x: {original_size} -> {new_size} px",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={
            "factor": factor,
            "original_size": original_size,
            "new_size": new_size,
        },
    )


async def _execute_crop(registry: ToolRegistry, box: list[float]) -> ToolResult:
    """Execute crop tool."""
    if len(box) != 4:
        raise ToolExecutionError(f"Crop requires [x1, y1, x2, y2], got {len(box)} values")

    # Validate types before normalization
    coord_names = ["x1", "y1", "x2", "y2"]
    for i, value in enumerate(box):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ToolExecutionError(
                f"Crop coordinates must be numbers, got {coord_names[i]}={value!r} "
                f"(type {type(value).__name__})"
            )

    # Auto-normalize pixel coords before range validation
    image = _require_image(registry)
    box = _maybe_normalize_box(box, image)

    for i, value in enumerate(box):
        if not 0 <= value <= 1:
            raise ToolExecutionError(
                f"Crop coordinates must be in range [0, 1]. Got {coord_names[i]}={value}"
            )
    x1, y1, x2, y2 = box
    original_size = image.size

    try:
        current, encoded = await _transform_and_encode(
            registry, lambda img: crop_image(img, (x1, y1, x2, y2))
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid crop region: {e}") from e

    new_size = current.size
    area_percentage = (x2 - x1) * (y2 - y1) * 100

    return ToolResult(
        tool_name="crop",
        description=f"Cropped to [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}] ({area_percentage:.1f}%)",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={
            "box": box,
            "original_size": original_size,
            "new_size": new_size,
            "area_percentage": round(area_percentage, 1),
        },
    )


async def _execute_contrast(registry: ToolRegistry, factor: float) -> ToolResult:
    """Execute contrast adjustment tool."""
    image = _require_image(registry)
    original_size = image.size

    try:
        _, encoded = await _transform_and_encode(registry, lambda img: adjust_contrast(img, factor))
    except ValueError as e:
        raise ToolExecutionError(f"Invalid contrast factor: {e}") from e

    return ToolResult(
        tool_name="adjust_contrast",
        description=f"Adjusted contrast by factor {factor:.1f}x",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"factor": factor, "size": original_size},
    )


async def _execute_threshold(registry: ToolRegistry, lower: int, upper: int) -> ToolResult:
    """Execute intensity threshold tool."""
    image = _require_image(registry)
    original_size = image.size
    was_modified = registry.get_image_manager().is_modified

    try:
        _, encoded = await _transform_and_encode(
            registry, lambda img: apply_intensity_threshold(img, lower, upper)
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid threshold bounds: {e}") from e

    desc = f"Applied threshold [{lower}, {upper}] (converts to grayscale)"
    if was_modified:
        desc += " WARNING: applied to already-modified image, not original intensities"
    return ToolResult(
        tool_name="threshold",
        description=desc,
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"lower": lower, "upper": upper, "size": original_size},
    )


async def _execute_flip_horizontal(registry: ToolRegistry) -> ToolResult:
    """Execute horizontal flip tool."""
    _require_image(registry)

    current, encoded = await _transform_and_encode(registry, flip_horizontal)

    return ToolResult(
        tool_name="flip_horizontal",
        description=(
            "Flipped image horizontally. "
            "WARNING — LATERALITY REVERSED: left and right are now swapped. "
            "All spatial references (e.g. 'left temporal') are mirrored. "
            "Call reset() before final answer to restore original orientation."
        ),
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"size": current.size, "laterality_reversed": True},
    )


async def _execute_flip_vertical(registry: ToolRegistry) -> ToolResult:
    """Execute vertical flip tool."""
    _require_image(registry)

    current, encoded = await _transform_and_encode(registry, flip_vertical)

    return ToolResult(
        tool_name="flip_vertical",
        description=(
            "Flipped image vertically. "
            "WARNING — ORIENTATION REVERSED: superior and inferior are now swapped. "
            "All spatial references (e.g. 'superior frontal') are mirrored. "
            "Call reset() before final answer to restore original orientation."
        ),
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"size": current.size, "orientation_reversed": True},
    )


async def _execute_rotate(registry: ToolRegistry, clockwise: bool = True) -> ToolResult:
    """Execute rotation tool."""
    image = _require_image(registry)
    original_size = image.size

    current, encoded = await _transform_and_encode(
        registry, lambda img: rotate_90(img, clockwise=clockwise)
    )

    direction = "clockwise" if clockwise else "counter-clockwise"
    return ToolResult(
        tool_name="rotate",
        description=f"Rotated 90° {direction}",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={
            "clockwise": clockwise,
            "original_size": original_size,
            "new_size": current.size,
        },
    )


async def _execute_reset(registry: ToolRegistry) -> ToolResult:
    """Reset to original image."""
    image_manager = registry.get_image_manager()
    if image_manager.image_path is None:
        raise ToolExecutionError("Cannot reset: no original image path stored")

    image_manager.reset_to_original()

    current = _get_current_image(registry)

    # Reuse cached encoding of the original image when available
    encoded = image_manager.original_encoding
    if encoded is None:
        encoded = await asyncio.to_thread(encode_image, current)
        image_manager.original_encoding = encoded

    return ToolResult(
        tool_name="reset",
        description="Reset to original image",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"size": current.size},
    )


async def _execute_brightness(registry: ToolRegistry, factor: float) -> ToolResult:
    """Execute brightness adjustment tool."""
    image = _require_image(registry)
    original_size = image.size

    try:
        _, encoded = await _transform_and_encode(
            registry, lambda img: adjust_brightness(img, factor)
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid brightness factor: {e}") from e

    return ToolResult(
        tool_name="adjust_brightness",
        description=f"Adjusted brightness by factor {factor:.1f}x",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"factor": factor, "size": original_size},
    )


async def _execute_sharpness(registry: ToolRegistry, factor: float) -> ToolResult:
    """Execute sharpness adjustment tool."""
    image = _require_image(registry)
    original_size = image.size

    try:
        _, encoded = await _transform_and_encode(
            registry, lambda img: adjust_sharpness(img, factor)
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid sharpness factor: {e}") from e

    return ToolResult(
        tool_name="adjust_sharpness",
        description=f"Adjusted sharpness by factor {factor:.1f}x",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"factor": factor, "size": original_size},
    )


async def _execute_equalize(registry: ToolRegistry) -> ToolResult:
    """Execute histogram equalization tool."""
    _require_image(registry)

    current, encoded = await _transform_and_encode(registry, equalize_histogram)

    return ToolResult(
        tool_name="equalize_histogram",
        description="Applied histogram equalization",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"size": current.size},
    )


async def _execute_intensity_stats(
    registry: ToolRegistry, box: list[float] | str | None = None
) -> ToolResult:
    """Execute intensity statistics tool."""
    image = _require_image(registry)

    # Models sometimes pass the string "null" instead of JSON null.
    # Any other string (e.g. "[0.1,0.2,0.3,0.4]") is a model error —
    # silently treating it as None would return wrong (full-image) stats.
    if isinstance(box, str):
        if box.lower() in ("null", "none", ""):
            box = None
        else:
            raise ToolExecutionError(
                f"box must be an array [x1,y1,x2,y2] or null, got string: {box!r}"
            )

    box_tuple: tuple[float, float, float, float] | None = None
    if box is not None:
        if len(box) != 4:
            raise ToolExecutionError(f"box requires [x1, y1, x2, y2], got {len(box)} values")
        box = _maybe_normalize_box(box, image)
        for i, value in enumerate(box):
            if not 0 <= value <= 1:
                raise ToolExecutionError(
                    f"box coordinates must be in [0, 1], got value {value} at index {i}"
                )
        box_tuple = (box[0], box[1], box[2], box[3])

    try:
        stats = await asyncio.to_thread(get_intensity_stats, image, box_tuple)
    except ValueError as e:
        raise ToolExecutionError(f"Invalid box coordinates: {e}") from e

    region = f" for region {box}" if box is not None else ""
    desc = f"Computed intensity statistics{region}"
    if registry.get_image_manager().is_modified:
        desc += " (on modified image — use 'reset' for original values)"
    stats["image_size"] = image.size
    return ToolResult(
        tool_name="get_intensity_stats",
        description=desc,
        image_base64=None,
        image_mime_type=None,
        metadata=stats,
    )


async def _execute_measure(
    registry: ToolRegistry, point1: list[float], point2: list[float]
) -> ToolResult:
    """Execute distance measurement tool."""
    # Validate length before normalization to avoid silent truncation
    for name, pt in [("point1", point1), ("point2", point2)]:
        if len(pt) != 2:
            raise ToolExecutionError(f"{name} must have 2 values [x, y], got {len(pt)}")

    image = _require_image(registry)
    point1 = _maybe_normalize_point(point1, image)
    point2 = _maybe_normalize_point(point2, image)

    for name, pt in [("point1", point1), ("point2", point2)]:
        for i, value in enumerate(pt):
            if not 0 <= value <= 1:
                raise ToolExecutionError(
                    f"{name} coordinates must be in [0, 1], got value {value} at index {i}"
                )

    try:
        result = await asyncio.to_thread(
            measure_distance, image, (point1[0], point1[1]), (point2[0], point2[1])
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid measurement points: {e}") from e

    dist = result["distance_pixels"]
    desc = f"Measured distance: {dist:.1f} pixels"
    if registry.get_image_manager().is_modified:
        desc += " (on modified image — use 'reset' for original-space measurement)"
    return ToolResult(
        tool_name="measure",
        description=desc,
        image_base64=None,
        image_mime_type=None,
        metadata=result,
    )


async def _execute_show_grid(registry: ToolRegistry, divisions: int) -> ToolResult:
    """Execute grid overlay tool (read-only — does not mutate image state)."""
    _require_image(registry)

    try:
        current, encoded = await _read_only_encode(
            registry, lambda img: draw_grid_overlay(img, divisions)
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid grid divisions: {e}") from e

    # Build cell labels list
    cell_labels = [
        f"{chr(65 + col)}{row + 1}" for row in range(divisions) for col in range(divisions)
    ]

    return ToolResult(
        tool_name="show_grid",
        description=f"Applied {divisions}x{divisions} grid overlay (visual only, image unchanged)",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"divisions": divisions, "cell_labels": cell_labels, "size": current.size},
    )


async def _execute_detect_edges(registry: ToolRegistry, method: str = "sobel") -> ToolResult:
    """Execute edge detection tool."""
    _require_image(registry)

    try:
        current, encoded = await _transform_and_encode(
            registry, lambda img: detect_edges(img, method)
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid edge detection method: {e}") from e

    return ToolResult(
        tool_name="detect_edges",
        description=f"Detected edges using {method} operator",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"method": method, "size": current.size},
    )


async def _execute_symmetry_diff(registry: ToolRegistry) -> ToolResult:
    """Execute symmetry difference tool."""
    _require_image(registry)

    current, encoded = await _transform_and_encode(registry, compute_symmetry_diff)

    return ToolResult(
        tool_name="symmetry_diff",
        description="Computed left-right symmetry difference map",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"size": current.size},
    )


async def _execute_annotate_region(
    registry: ToolRegistry,
    box: list[float],
    color: str = "red",
    label: str | None = None,
) -> ToolResult:
    """Execute region annotation tool."""
    if len(box) != 4:
        raise ToolExecutionError(f"box requires [x1, y1, x2, y2], got {len(box)} values")

    # Validate element types before normalization (same guard as _execute_crop)
    coord_names = ["x1", "y1", "x2", "y2"]
    for i, value in enumerate(box):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ToolExecutionError(
                f"box coordinates must be numbers, got {coord_names[i]}={value!r} "
                f"(type {type(value).__name__})"
            )

    image = _require_image(registry)
    box = _maybe_normalize_box(box, image)

    for i, value in enumerate(box):
        if not 0 <= value <= 1:
            raise ToolExecutionError(
                f"box coordinates must be in [0, 1], got value {value} at index {i}"
            )
    box_tuple = (box[0], box[1], box[2], box[3])

    try:
        current, encoded = await _read_only_encode(
            registry, lambda img: annotate_region(img, box_tuple, color=color, label=label)
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid annotation: {e}") from e

    return ToolResult(
        tool_name="annotate_region",
        description=(
            f"Annotated region [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}] "
            f"(visual only, image unchanged)"
        ),
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"box": box, "color": color, "label": label, "size": current.size},
    )


async def _execute_invert(registry: ToolRegistry) -> ToolResult:
    """Execute image inversion tool."""
    _require_image(registry)

    current, encoded = await _transform_and_encode(registry, invert_image)

    return ToolResult(
        tool_name="invert",
        description="Inverted image intensities",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"size": current.size},
    )


async def _execute_window_level(
    registry: ToolRegistry,
    center: int | None = None,
    width: int | None = None,
    preset: str | None = None,
) -> ToolResult:
    """Execute clinical window/level tool."""
    _require_image(registry)
    was_modified = registry.get_image_manager().is_modified

    try:
        current, encoded = await _transform_and_encode(
            registry,
            lambda img: apply_window_level(img, center=center, width=width, preset=preset),
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid window/level: {e}") from e

    # Report the *actual* values used (preset overrides kwargs).
    actual_center = center
    actual_width = width
    if preset is not None and preset in WINDOW_PRESETS:
        actual_center, actual_width = WINDOW_PRESETS[preset]

    desc = (
        f"Applied window preset '{preset}' (center={actual_center}, width={actual_width})"
        if preset
        else f"Applied window center={center} width={width}"
    )
    desc += " (converts to grayscale)"
    if was_modified:
        desc += " WARNING: applied to already-modified image, not original intensities"
    return ToolResult(
        tool_name="window_level",
        description=desc,
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={
            "center": actual_center,
            "width": actual_width,
            "preset": preset,
            "size": current.size,
        },
    )


async def _execute_adaptive_equalize(
    registry: ToolRegistry,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> ToolResult:
    """Execute CLAHE tool."""
    _require_image(registry)

    try:
        current, encoded = await _transform_and_encode(
            registry,
            lambda img: adaptive_equalize(img, clip_limit=clip_limit, tile_size=tile_size),
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid CLAHE parameters: {e}") from e

    return ToolResult(
        tool_name="adaptive_equalize",
        description=f"Applied CLAHE (clip={clip_limit}, tiles={tile_size})",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"clip_limit": clip_limit, "tile_size": tile_size, "size": current.size},
    )


async def _execute_intensity_profile(
    registry: ToolRegistry, point1: list[float], point2: list[float]
) -> ToolResult:
    """Execute intensity profile tool."""
    # Validate length before normalization to avoid silent truncation
    for name, pt in [("point1", point1), ("point2", point2)]:
        if len(pt) != 2:
            raise ToolExecutionError(f"{name} must have 2 values [x, y], got {len(pt)}")

    image = _require_image(registry)
    point1 = _maybe_normalize_point(point1, image)
    point2 = _maybe_normalize_point(point2, image)

    for name, pt in [("point1", point1), ("point2", point2)]:
        for i, value in enumerate(pt):
            if not 0 <= value <= 1:
                raise ToolExecutionError(
                    f"{name} coordinates must be in [0, 1], got value {value} at index {i}"
                )

    try:
        result = await asyncio.to_thread(
            compute_intensity_profile, image, (point1[0], point1[1]), (point2[0], point2[1])
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid profile points: {e}") from e

    desc = f"Sampled {result['n_samples']} points along line"
    if registry.get_image_manager().is_modified:
        desc += " (on modified image — use 'reset' for original intensities)"
    result["image_size"] = image.size
    return ToolResult(
        tool_name="intensity_profile",
        description=desc,
        image_base64=None,
        image_mime_type=None,
        metadata=result,
    )


async def _execute_denoise(registry: ToolRegistry, sigma: float) -> ToolResult:
    """Execute Gaussian denoise tool."""
    _require_image(registry)

    try:
        current, encoded = await _transform_and_encode(
            registry, lambda img: denoise_gaussian(img, sigma)
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid denoise sigma: {e}") from e

    return ToolResult(
        tool_name="denoise",
        description=f"Applied Gaussian denoise (sigma={sigma})",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"sigma": sigma, "size": current.size},
    )


async def _execute_morphological(
    registry: ToolRegistry,
    operation: str,
    iterations: int = 1,
    threshold_value: int | None = None,
) -> ToolResult:
    """Execute morphological operation tool."""
    _require_image(registry)

    try:
        current, encoded = await _transform_and_encode(
            registry,
            lambda img: morphological_op(
                img, operation, iterations=iterations, threshold_value=threshold_value
            ),
        )
    except ValueError as e:
        raise ToolExecutionError(f"Invalid morphological parameters: {e}") from e

    desc = f"Applied morphological {operation} x{iterations}"
    if threshold_value is not None:
        desc += f" (threshold={threshold_value})"
    return ToolResult(
        tool_name="morphological",
        description=desc,
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={
            "operation": operation,
            "iterations": iterations,
            "threshold_value": threshold_value,
            "size": current.size,
        },
    )


# Static prompt documentation for tools with fixed parameters
_CROP_PROMPT_DOC = (
    "**crop** - Extract region [x1, y1, x2, y2] normalized (0-1); pixel coords auto-converted"
)

_THRESHOLD_PROMPT_DOC = (
    "**threshold** - Apply intensity windowing, converts to grayscale (lower: 0-254, upper: 1-255)"
)

_FLIP_HORIZONTAL_PROMPT_DOC = "**flip_horizontal** - Mirror image left-right"

_FLIP_VERTICAL_PROMPT_DOC = "**flip_vertical** - Mirror image top-bottom"

_ROTATE_PROMPT_DOC = "**rotate** - Rotate image by 90 degrees (clockwise: boolean, default true)"

_RESET_PROMPT_DOC = "**reset** - Return to original image"

_EQUALIZE_PROMPT_DOC = "**equalize_histogram** - Equalize intensity distribution (grayscale)"

_INTENSITY_STATS_PROMPT_DOC = (
    "**get_intensity_stats** - Get intensity statistics "
    "(optional box [x1,y1,x2,y2] 0-1; pixel coords auto-converted)"
)

_MEASURE_PROMPT_DOC = (
    "**measure** - Measure distance between two points "
    "(point1, point2: [x,y] 0-1; pixel coords auto-converted)"
)

_SYMMETRY_DIFF_PROMPT_DOC = (
    "**symmetry_diff** - Compute left-right symmetry difference map (converts to grayscale)"
)

_INVERT_PROMPT_DOC = "**invert** - Invert image intensities, converts to grayscale (negative)"

_INTENSITY_PROFILE_PROMPT_DOC = (
    "**intensity_profile** - Sample intensities along a line "
    "(point1, point2: [x,y] 0-1; pixel coords auto-converted)"
)

_ANNOTATE_REGION_PROMPT_DOC = (
    "**annotate_region** - Draw bounding box overlay, "
    "visual only (box [x1,y1,x2,y2] 0-1, color, label)"
)


@beartype
def create_visual_tools(
    disabled_tools: set[str] | None = None,
    config: ImageProcessingConfig | None = None,
) -> list[Tool]:
    """Create the standard set of visual tools for image analysis.

    Args:
        disabled_tools: Set of tool names to exclude from the returned list
        config: Image processing config for schema ranges. If None, uses global default.

    Returns:
        List of Tool objects ready for registration with ToolRegistry
    """
    cfg = config or _get_image_config()
    disabled = disabled_tools or set()
    tools: list[Tool] = []

    # Generate prompt docs from config so ranges stay in sync
    zoom_prompt_doc = (
        f"**zoom** - Magnify the image (factor: {cfg.min_zoom_factor}-{cfg.max_zoom_factor})"
    )
    contrast_prompt_doc = (
        f"**adjust_contrast** - Enhance contrast "
        f"(factor: {cfg.min_contrast_factor}-{cfg.max_contrast_factor})"
    )

    if "zoom" not in disabled:
        tools.append(
            Tool(
                name="zoom",
                description="Magnify image for detail analysis. Use 2.0-4.0 factor.",
                parameters={
                    "factor": {
                        "type": "number",
                        "description": "Zoom factor: 1.0=unchanged, 2.0=2x zoom.",
                        "minimum": cfg.min_zoom_factor,
                        "maximum": cfg.max_zoom_factor,
                    }
                },
                execute=_execute_zoom,
                requires_image=True,
                prompt_documentation=zoom_prompt_doc,
                category="visual",
            )
        )

    if "crop" not in disabled:
        tools.append(
            Tool(
                name="crop",
                description="Extract rectangular region for focused analysis.",
                parameters={
                    "box": {
                        "type": "array",
                        "description": "Box [x1,y1,x2,y2] normalized (0-1) coordinates.",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 4,
                        "maxItems": 4,
                    }
                },
                execute=_execute_crop,
                requires_image=True,
                prompt_documentation=_CROP_PROMPT_DOC,
                category="visual",
            )
        )

    if "adjust_contrast" not in disabled:
        tools.append(
            Tool(
                name="adjust_contrast",
                description=(
                    "Enhance or reduce image contrast to better distinguish tissue "
                    "boundaries and subtle findings."
                ),
                parameters={
                    "factor": {
                        "type": "number",
                        "description": (
                            "Contrast factor: 1.0=no change, >1.0=increase, <1.0=decrease."
                        ),
                        "minimum": cfg.min_contrast_factor,
                        "maximum": cfg.max_contrast_factor,
                    }
                },
                execute=_execute_contrast,
                requires_image=True,
                prompt_documentation=contrast_prompt_doc,
                category="visual",
            )
        )

    if "adjust_brightness" not in disabled:
        brightness_prompt_doc = (
            f"**adjust_brightness** - Adjust brightness "
            f"(factor: {cfg.min_brightness_factor}-{cfg.max_brightness_factor})"
        )
        tools.append(
            Tool(
                name="adjust_brightness",
                description=(
                    "Adjust image brightness (window level). "
                    "1.0 = no change, >1.0 = brighter, <1.0 = darker."
                ),
                parameters={
                    "factor": {
                        "type": "number",
                        "description": (
                            "Brightness factor: 1.0=no change, >1.0=brighter, <1.0=darker."
                        ),
                        "minimum": cfg.min_brightness_factor,
                        "maximum": cfg.max_brightness_factor,
                    }
                },
                execute=_execute_brightness,
                requires_image=True,
                prompt_documentation=brightness_prompt_doc,
                category="visual",
            )
        )

    if "adjust_sharpness" not in disabled:
        sharpness_prompt_doc = (
            f"**adjust_sharpness** - Adjust sharpness "
            f"(factor: {cfg.min_sharpness_factor}-{cfg.max_sharpness_factor})"
        )
        tools.append(
            Tool(
                name="adjust_sharpness",
                description=(
                    "Adjust image sharpness. 0.0 = blurred, 1.0 = original, >1.0 = sharpened."
                ),
                parameters={
                    "factor": {
                        "type": "number",
                        "description": "Sharpness factor: 0.0=blurred, 1.0=original, >1.0=sharper.",
                        "minimum": cfg.min_sharpness_factor,
                        "maximum": cfg.max_sharpness_factor,
                    }
                },
                execute=_execute_sharpness,
                requires_image=True,
                prompt_documentation=sharpness_prompt_doc,
                category="visual",
            )
        )

    if "threshold" not in disabled:
        tools.append(
            Tool(
                name="threshold",
                description=(
                    "Apply intensity windowing to highlight specific tissue types or abnormalities."
                ),
                parameters={
                    "lower": {
                        "type": "integer",
                        "description": "Lower intensity bound (0-254).",
                        "minimum": 0,
                        "maximum": 254,
                    },
                    "upper": {
                        "type": "integer",
                        "description": "Upper intensity bound (1-255). Must be > lower.",
                        "minimum": 1,
                        "maximum": 255,
                    },
                },
                execute=_execute_threshold,
                requires_image=True,
                prompt_documentation=_THRESHOLD_PROMPT_DOC,
                category="visual",
            )
        )

    if "window_level" not in disabled:
        presets_list = ", ".join(sorted(WINDOW_PRESETS.keys()))
        window_prompt_doc = (
            f"**window_level** - Clinical windowing, converts to grayscale. "
            f"Presets assume CT Hounsfield units; use center/width for 8-bit images. "
            f"Must provide EITHER preset ({presets_list}) OR both center and width."
        )
        tools.append(
            Tool(
                name="window_level",
                description=(
                    "Apply clinical window/level settings. "
                    "Use preset names ("
                    + ", ".join(sorted(WINDOW_PRESETS.keys()))
                    + ") or specify center and width."
                ),
                parameters={
                    "center": {
                        "type": "integer",
                        "description": "Window center intensity. Required if no preset.",
                        "default": None,
                    },
                    "width": {
                        "type": "integer",
                        "description": "Window width. Required if no preset.",
                        "default": None,
                    },
                    "preset": {
                        "type": "string",
                        "description": "Clinical window preset name.",
                        "enum": sorted(WINDOW_PRESETS.keys()),
                        "default": None,
                    },
                },
                execute=_execute_window_level,
                requires_image=True,
                prompt_documentation=window_prompt_doc,
                category="visual",
            )
        )

    if "equalize_histogram" not in disabled:
        tools.append(
            Tool(
                name="equalize_histogram",
                description=(
                    "Equalize intensity histogram for improved contrast distribution (grayscale)."
                ),
                parameters={},
                execute=_execute_equalize,
                requires_image=True,
                prompt_documentation=_EQUALIZE_PROMPT_DOC,
                category="visual",
            )
        )

    if "adaptive_equalize" not in disabled:
        clahe_prompt_doc = (
            f"**adaptive_equalize** - CLAHE local contrast "
            f"(clip_limit: {cfg.min_clahe_clip_limit}-{cfg.max_clahe_clip_limit})"
        )
        tools.append(
            Tool(
                name="adaptive_equalize",
                description=(
                    "Contrast Limited Adaptive Histogram Equalization (CLAHE). "
                    "Better local contrast than global equalization."
                ),
                parameters={
                    "clip_limit": {
                        "type": "number",
                        "description": "Histogram clip limit (higher = more contrast).",
                        "minimum": cfg.min_clahe_clip_limit,
                        "maximum": cfg.max_clahe_clip_limit,
                        "default": 2.0,
                    },
                    "tile_size": {
                        "type": "integer",
                        "description": "Tile grid size for local processing.",
                        "minimum": 2,
                        "maximum": cfg.max_clahe_tile_size,
                        "default": 8,
                    },
                },
                execute=_execute_adaptive_equalize,
                requires_image=True,
                prompt_documentation=clahe_prompt_doc,
                category="visual",
            )
        )

    if "detect_edges" not in disabled:
        tools.append(
            Tool(
                name="detect_edges",
                description=(
                    "Detect edges using Sobel or Laplacian operators for boundary delineation."
                ),
                parameters={
                    "method": {
                        "type": "string",
                        "description": "Edge detection method.",
                        "enum": ["sobel", "laplacian"],
                        "default": "sobel",
                    }
                },
                execute=_execute_detect_edges,
                requires_image=True,
                prompt_documentation=(
                    "**detect_edges** - Edge detection, converts "
                    "to grayscale (method: sobel/laplacian)"
                ),
                category="visual",
            )
        )

    if "denoise" not in disabled:
        denoise_prompt_doc = (
            f"**denoise** - Gaussian noise reduction "
            f"(sigma: {cfg.min_gaussian_sigma}-{cfg.max_gaussian_sigma})"
        )
        tools.append(
            Tool(
                name="denoise",
                description="Apply Gaussian blur for noise reduction.",
                parameters={
                    "sigma": {
                        "type": "number",
                        "description": "Gaussian kernel sigma (higher = more smoothing).",
                        "minimum": cfg.min_gaussian_sigma,
                        "maximum": cfg.max_gaussian_sigma,
                    }
                },
                execute=_execute_denoise,
                requires_image=True,
                prompt_documentation=denoise_prompt_doc,
                category="visual",
            )
        )

    if "morphological" not in disabled:
        morph_prompt_doc = (
            f"**morphological** - Morphological ops, converts to grayscale "
            f"(operation: erode/dilate/open/close, "
            f"iterations: 1-{cfg.max_morphological_iterations})"
        )
        tools.append(
            Tool(
                name="morphological",
                description=(
                    "Morphological operations for mask cleanup after thresholding. "
                    "erode=shrink, dilate=expand, open=remove noise, close=fill holes."
                ),
                parameters={
                    "operation": {
                        "type": "string",
                        "description": "Morphological operation to apply.",
                        "enum": ["erode", "dilate", "open", "close"],
                    },
                    "iterations": {
                        "type": "integer",
                        "description": "Number of iterations.",
                        "minimum": 1,
                        "maximum": cfg.max_morphological_iterations,
                        "default": 1,
                    },
                    "threshold_value": {
                        "type": "integer",
                        "description": "Optional: binarize at this intensity first (0-255).",
                        "minimum": 0,
                        "maximum": 255,
                        "default": None,
                    },
                },
                execute=_execute_morphological,
                requires_image=True,
                prompt_documentation=morph_prompt_doc,
                category="visual",
            )
        )

    if "get_intensity_stats" not in disabled:
        tools.append(
            Tool(
                name="get_intensity_stats",
                description=(
                    "Compute intensity statistics (mean, std, min, max, median, histogram) "
                    "over the full image or a sub-region."
                ),
                parameters={
                    "box": {
                        "type": "array",
                        "description": "Optional [x1,y1,x2,y2] normalized (0-1) sub-region.",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 4,
                        "maxItems": 4,
                        "default": None,
                    }
                },
                execute=_execute_intensity_stats,
                requires_image=True,
                prompt_documentation=_INTENSITY_STATS_PROMPT_DOC,
                category="visual",
            )
        )

    if "intensity_profile" not in disabled:
        tools.append(
            Tool(
                name="intensity_profile",
                description=(
                    "Sample pixel intensities along a line between two points. "
                    "Differentiates cyst (sharp drop) vs tumor (gradual) vs edema."
                ),
                parameters={
                    "point1": {
                        "type": "array",
                        "description": "Start point [x, y] in normalized (0-1) coordinates.",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "point2": {
                        "type": "array",
                        "description": "End point [x, y] in normalized (0-1) coordinates.",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                execute=_execute_intensity_profile,
                requires_image=True,
                prompt_documentation=_INTENSITY_PROFILE_PROMPT_DOC,
                category="visual",
            )
        )

    if "symmetry_diff" not in disabled:
        tools.append(
            Tool(
                name="symmetry_diff",
                description=(
                    "Compute left-right symmetry difference map. "
                    "Bright regions indicate asymmetry (potential pathology)."
                ),
                parameters={},
                execute=_execute_symmetry_diff,
                requires_image=True,
                prompt_documentation=_SYMMETRY_DIFF_PROMPT_DOC,
                category="visual",
            )
        )

    if "invert" not in disabled:
        tools.append(
            Tool(
                name="invert",
                description=(
                    "Invert pixel intensities (negative image). "
                    "Toggling display mode can reveal findings hidden in one polarity."
                ),
                parameters={},
                execute=_execute_invert,
                requires_image=True,
                prompt_documentation=_INVERT_PROMPT_DOC,
                category="visual",
            )
        )

    if "annotate_region" not in disabled:
        tools.append(
            Tool(
                name="annotate_region",
                description=(
                    "Draw a bounding box overlay on the image for propose-and-verify localization."
                ),
                parameters={
                    "box": {
                        "type": "array",
                        "description": "Box [x1,y1,x2,y2] normalized (0-1) coordinates.",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "color": {
                        "type": "string",
                        "description": "Box outline color.",
                        "enum": ["red", "green", "yellow", "blue", "white", "cyan", "magenta"],
                        "default": "red",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional text label above the box.",
                        "default": None,
                    },
                },
                execute=_execute_annotate_region,
                requires_image=True,
                prompt_documentation=_ANNOTATE_REGION_PROMPT_DOC,
                category="visual",
            )
        )

    if "flip_horizontal" not in disabled:
        tools.append(
            Tool(
                name="flip_horizontal",
                description="Mirror the image left-right for bilateral symmetry assessment.",
                parameters={},
                execute=_execute_flip_horizontal,
                requires_image=True,
                prompt_documentation=_FLIP_HORIZONTAL_PROMPT_DOC,
                category="visual",
            )
        )

    if "flip_vertical" not in disabled:
        tools.append(
            Tool(
                name="flip_vertical",
                description="Mirror the image top-bottom for orientation standardization.",
                parameters={},
                execute=_execute_flip_vertical,
                requires_image=True,
                prompt_documentation=_FLIP_VERTICAL_PROMPT_DOC,
                category="visual",
            )
        )

    if "rotate" not in disabled:
        tools.append(
            Tool(
                name="rotate",
                description="Rotate image by 90 degrees clockwise or counter-clockwise.",
                parameters={
                    "clockwise": {
                        "type": "boolean",
                        "description": "If true, rotate clockwise; if false, counter-clockwise.",
                        "default": True,
                    }
                },
                execute=_execute_rotate,
                requires_image=True,
                prompt_documentation=_ROTATE_PROMPT_DOC,
                category="visual",
            )
        )

    if "show_grid" not in disabled:
        grid_prompt_doc = (
            "**show_grid** - Overlay labeled grid, visual only "
            f"(divisions: 2-{cfg.max_grid_divisions})"
        )
        tools.append(
            Tool(
                name="show_grid",
                description="Overlay a labeled grid for spatial reference (e.g. A1, B2).",
                parameters={
                    "divisions": {
                        "type": "integer",
                        "description": "Grid divisions per axis (rows and columns).",
                        "minimum": 2,
                        "maximum": cfg.max_grid_divisions,
                    }
                },
                execute=_execute_show_grid,
                requires_image=True,
                prompt_documentation=grid_prompt_doc,
                category="visual",
            )
        )

    if "measure" not in disabled:
        tools.append(
            Tool(
                name="measure",
                description="Measure Euclidean distance between two points in pixels.",
                parameters={
                    "point1": {
                        "type": "array",
                        "description": "First point [x, y] in normalized (0-1) coordinates.",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "point2": {
                        "type": "array",
                        "description": "Second point [x, y] in normalized (0-1) coordinates.",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                execute=_execute_measure,
                requires_image=True,
                prompt_documentation=_MEASURE_PROMPT_DOC,
                category="visual",
            )
        )

    if "reset" not in disabled:
        tools.append(
            Tool(
                name="reset",
                description="Return to the original full image, discarding all modifications.",
                parameters={},
                execute=_execute_reset,
                requires_image=True,
                prompt_documentation=_RESET_PROMPT_DOC,
                category="visual",
            )
        )

    return tools
