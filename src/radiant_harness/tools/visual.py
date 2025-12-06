"""Visual tools for radiology image analysis.

Provides zoom, crop, contrast adjustment, thresholding, flip, and rotate tools
for interactive image analysis by VLMs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from beartype import beartype
from PIL import Image

from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.tools.image_ops import adjust_contrast
from radiant_harness.tools.image_ops import apply_intensity_threshold
from radiant_harness.tools.image_ops import crop_image
from radiant_harness.tools.image_ops import flip_horizontal
from radiant_harness.tools.image_ops import flip_vertical
from radiant_harness.tools.image_ops import rotate_90
from radiant_harness.tools.image_ops import zoom_image
from radiant_harness.tools.registry import Tool
from radiant_harness.tools.registry import encode_image
from radiant_harness.types import ToolResult

if TYPE_CHECKING:
    from radiant_harness.tools.registry import ToolRegistry


def _require_image(registry: ToolRegistry) -> Image.Image:
    """Get current image from registry, raising if none is loaded."""
    if registry.current_image is None:
        raise ToolExecutionError("Tool requires a loaded image but none is active")
    return registry.current_image


def _get_current_image(registry: ToolRegistry) -> Image.Image:
    """Get current image after a transform (guaranteed non-None by transform_image)."""
    img = registry.current_image
    if img is None:
        # This should never happen after transform_image, but satisfies type checker
        raise ToolExecutionError("Image unexpectedly None after transform")
    return img


async def _execute_zoom(registry: ToolRegistry, factor: float) -> ToolResult:
    """Execute zoom tool."""
    image = _require_image(registry)
    original_size = image.size

    try:
        registry.transform_image(lambda img: zoom_image(img, factor))
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

    image = _require_image(registry)
    x1, y1, x2, y2 = box
    original_size = image.size

    try:
        registry.transform_image(lambda img: crop_image(img, (x1, y1, x2, y2)))
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

    try:
        registry.transform_image(lambda img: adjust_contrast(img, factor))
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

    try:
        registry.transform_image(lambda img: apply_intensity_threshold(img, lower, upper))
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
    registry.transform_image(flip_horizontal)

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
    registry.transform_image(flip_vertical)

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

    registry.transform_image(lambda img: rotate_90(img, clockwise=clockwise))

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
    if registry.image_path is None:
        raise ToolExecutionError("Cannot reset: no original image path stored")

    with Image.open(registry.image_path) as img:
        registry.current_image = img.copy()

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
ZOOM_PROMPT_DOC = """
**zoom** - Magnify the image for detailed examination
  - Parameter `factor` (number, 0.5-4.0): Magnification level (2.0 = 2x zoom)
  - Use for: Examining small lesions, tissue boundaries, subtle findings
  - Recommended factors: 1.5x for overview, 2.0-2.5x for detail, 3.0+ for micro-features
""".strip()

CROP_PROMPT_DOC = """
**crop** - Extract a specific region for focused analysis
  - Parameter `box` (array of 4 numbers, 0-1): Normalized coordinates [x1, y1, x2, y2]
  - Coordinates are normalized: 0.0 = left/top edge, 1.0 = right/bottom edge
  - Use for: Isolating anatomical regions, focusing on specific findings
  - Keep context: recommend 0.1-0.2 margin around region of interest
""".strip()

CONTRAST_PROMPT_DOC = """
**adjust_contrast** - Enhance image contrast for better visualization
  - Parameter `factor` (number, 0.5-3.0): Contrast multiplier (1.0 = no change)
  - Use for: Distinguishing tissue boundaries, detecting low-contrast lesions
  - Recommended: 1.3-1.5 for subtle enhancement, 1.8-2.0 for significant enhancement
""".strip()

THRESHOLD_PROMPT_DOC = """
**threshold** - Apply intensity windowing to isolate specific intensity ranges
  - Parameter `lower` (integer, 0-254): Lower intensity bound
  - Parameter `upper` (integer, 1-255): Upper intensity bound (must be > lower)
  - Pixels below lower become black, above upper become white, between are rescaled
  - Use for: Highlighting specific tissue types, isolating signal abnormalities
""".strip()

FLIP_HORIZONTAL_PROMPT_DOC = """
**flip_horizontal** - Mirror image left-right (no parameters)
  - Use for: Assessing bilateral symmetry, comparing hemispheres
  - Helpful for: Detecting asymmetric pathology, midline structure analysis
""".strip()

FLIP_VERTICAL_PROMPT_DOC = """
**flip_vertical** - Mirror image top-bottom (no parameters)
  - Use for: Evaluating vertical relationships, orientation analysis
  - Helpful for: Superior-inferior comparisons, anatomical orientation
""".strip()

ROTATE_PROMPT_DOC = """
**rotate** - Rotate image by 90 degrees
  - Parameter `clockwise` (boolean, default true): Direction of rotation
  - Use for: Standardizing orientation, examining from different angles
""".strip()

RESET_PROMPT_DOC = """
**reset** - Return to original unmodified image (no parameters)
  - Use for: Starting fresh after modifications, final verification
  - Discards all zoom, crop, contrast, threshold, flip, and rotate changes
""".strip()


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
                prompt_documentation=CONTRAST_PROMPT_DOC,
                category="visual",
            )
        )

    if "threshold" not in disabled:
        tools.append(
            Tool(
                name="threshold",
                description=(
                    "Apply intensity windowing to highlight specific tissue types "
                    "or abnormalities."
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
                prompt_documentation=RESET_PROMPT_DOC,
                category="visual",
            )
        )

    return tools
