"""Visual tools for radiology image analysis.

Provides zoom, crop, contrast adjustment, thresholding, flip, and rotate tools
for interactive image analysis by VLMs.

This module combines:
- Core image operations (zoom, crop, contrast, threshold, flip, rotate)
- Tool wrappers for VLM integration
- Tool factory (create_visual_tools)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from beartype import beartype
from PIL import Image
from PIL import ImageEnhance

from radiant_harness.config import ImageProcessingConfig
from radiant_harness.config import get_config
from radiant_harness.exceptions import ToolExecutionError
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
def apply_intensity_threshold(image: Image.Image, lower: int, upper: int) -> Image.Image:
    """Apply intensity threshold to grayscale image and rescale to 0-255.

    Args:
        image: Input PIL Image
        lower: Lower intensity bound (0-254)
        upper: Upper intensity bound (must be > lower, max 255)

    Returns:
        Thresholded grayscale image with intensities rescaled to 0-255

    Raises:
        ValueError: If bounds are invalid (lower < 0, upper > 255, or upper <= lower)
    """
    if lower < 0:
        raise ValueError(f"lower must be >= 0, got {lower}")
    if upper > 255:
        raise ValueError(f"upper must be <= 255, got {upper}")
    if upper <= lower:
        raise ValueError(f"upper must be > lower, got lower={lower}, upper={upper}")

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


# =============================================================================
# Tool Executors
# =============================================================================


def _require_image(registry: ToolRegistry) -> Image.Image:
    """Get current image from registry, raising if none is loaded."""
    image_manager = registry.get_image_manager()
    if image_manager.current_image is None:
        raise ToolExecutionError("Tool requires a loaded image but none is active")
    return image_manager.current_image


def _get_current_image(registry: ToolRegistry) -> Image.Image:
    """Get current image after a transform (guaranteed non-None by transform_image)."""
    image_manager = registry.get_image_manager()
    img = image_manager.current_image
    if img is None:
        raise ToolExecutionError("Image unexpectedly None after transform")
    return img


async def _execute_zoom(registry: ToolRegistry, factor: float) -> ToolResult:
    """Execute zoom tool."""
    image = _require_image(registry)
    original_size = image.size
    image_manager = registry.get_image_manager()

    try:
        image_manager.transform_image(lambda img: zoom_image(img, factor))
    except ValueError as e:
        raise ToolExecutionError(f"Invalid zoom factor: {e}") from e

    current = _get_current_image(registry)
    new_size = current.size
    encoded = encode_image(current)

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

    # Validate box values are within [0, 1] range
    for i, value in enumerate(box):
        if not 0 <= value <= 1:
            raise ToolExecutionError(
                f"Crop coordinates must be in range [0, 1]. "
                f"Got {['x1', 'y1', 'x2', 'y2'][i]}={value}"
            )

    image = _require_image(registry)
    x1, y1, x2, y2 = box
    original_size = image.size
    image_manager = registry.get_image_manager()

    try:
        image_manager.transform_image(lambda img: crop_image(img, (x1, y1, x2, y2)))
    except ValueError as e:
        raise ToolExecutionError(f"Invalid crop region: {e}") from e

    current = _get_current_image(registry)
    new_size = current.size
    area_percentage = (x2 - x1) * (y2 - y1) * 100
    encoded = encode_image(current)

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
    image_manager = registry.get_image_manager()

    try:
        image_manager.transform_image(lambda img: adjust_contrast(img, factor))
    except ValueError as e:
        raise ToolExecutionError(f"Invalid contrast factor: {e}") from e

    current = _get_current_image(registry)
    encoded = encode_image(current)

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
    image_manager = registry.get_image_manager()

    try:
        image_manager.transform_image(lambda img: apply_intensity_threshold(img, lower, upper))
    except ValueError as e:
        raise ToolExecutionError(f"Invalid threshold bounds: {e}") from e

    current = _get_current_image(registry)
    encoded = encode_image(current)

    return ToolResult(
        tool_name="threshold",
        description=f"Applied threshold [{lower}, {upper}]",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"lower": lower, "upper": upper, "size": original_size},
    )


async def _execute_flip_horizontal(registry: ToolRegistry) -> ToolResult:
    """Execute horizontal flip tool."""
    _require_image(registry)
    image_manager = registry.get_image_manager()
    image_manager.transform_image(flip_horizontal)

    current = _get_current_image(registry)
    encoded = encode_image(current)

    return ToolResult(
        tool_name="flip_horizontal",
        description="Flipped image horizontally",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"size": current.size},
    )


async def _execute_flip_vertical(registry: ToolRegistry) -> ToolResult:
    """Execute vertical flip tool."""
    _require_image(registry)
    image_manager = registry.get_image_manager()
    image_manager.transform_image(flip_vertical)

    current = _get_current_image(registry)
    encoded = encode_image(current)

    return ToolResult(
        tool_name="flip_vertical",
        description="Flipped image vertically",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"size": current.size},
    )


async def _execute_rotate(registry: ToolRegistry, clockwise: bool = True) -> ToolResult:
    """Execute rotation tool."""
    image = _require_image(registry)
    original_size = image.size
    image_manager = registry.get_image_manager()

    image_manager.transform_image(lambda img: rotate_90(img, clockwise=clockwise))

    current = _get_current_image(registry)
    direction = "clockwise" if clockwise else "counter-clockwise"
    encoded = encode_image(current)

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
    encoded = encode_image(current)

    return ToolResult(
        tool_name="reset",
        description="Reset to original image",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
        metadata={"size": current.size},
    )


# Prompt documentation for visual tools
ZOOM_PROMPT_DOC = """**zoom** - Magnify the image (factor: 0.5-4.0)""".strip()

CROP_PROMPT_DOC = (
    """**crop** - Extract region [x1, y1, x2, y2] with normalized coordinates (0-1)""".strip()
)

CONTRAST_PROMPT_DOC = """**adjust_contrast** - Enhance contrast (factor: 0.5-3.0)""".strip()

THRESHOLD_PROMPT_DOC = (
    """**threshold** - Apply intensity windowing (lower: 0-254, upper: 1-255)""".strip()
)

FLIP_HORIZONTAL_PROMPT_DOC = """**flip_horizontal** - Mirror image left-right""".strip()

FLIP_VERTICAL_PROMPT_DOC = """**flip_vertical** - Mirror image top-bottom""".strip()

ROTATE_PROMPT_DOC = (
    """**rotate** - Rotate image by 90 degrees (clockwise: boolean, default true)""".strip()
)

RESET_PROMPT_DOC = """**reset** - Return to original image""".strip()


@beartype
def create_visual_tools(disabled_tools: set[str] | None = None) -> list[Tool]:
    """Create the standard set of visual tools for image analysis.

    Args:
        disabled_tools: Set of tool names to exclude from the returned list

    Returns:
        List of Tool objects ready for registration with ToolRegistry
    """
    disabled = disabled_tools or set()
    tools: list[Tool] = []

    if "zoom" not in disabled:
        tools.append(
            Tool(
                name="zoom",
                description="Magnify image for detail analysis. Use 2.0-4.0 factor.",
                parameters={
                    "factor": {
                        "type": "number",
                        "description": "Zoom factor: 1.0=unchanged, 2.0=2x zoom.",
                        "minimum": 0.5,
                        "maximum": 4.0,
                    }
                },
                execute=_execute_zoom,
                requires_image=True,
                prompt_documentation=ZOOM_PROMPT_DOC,
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
                prompt_documentation=CROP_PROMPT_DOC,
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
                        "description": "Contrast factor: 1.0=no change, >1.0=increase, <1.0=decrease.",
                        "minimum": 0.5,
                        "maximum": 3.0,
                    }
                },
                execute=_execute_contrast,
                requires_image=True,
                prompt_documentation=CONTRAST_PROMPT_DOC,
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
                prompt_documentation=THRESHOLD_PROMPT_DOC,
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
                prompt_documentation=FLIP_HORIZONTAL_PROMPT_DOC,
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
                prompt_documentation=FLIP_VERTICAL_PROMPT_DOC,
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
                prompt_documentation=ROTATE_PROMPT_DOC,
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
                prompt_documentation=RESET_PROMPT_DOC,
                category="visual",
            )
        )

    return tools
