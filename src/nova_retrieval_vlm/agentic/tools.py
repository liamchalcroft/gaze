"""Visual tools for agentic medical image analysis.

Provides a registry of tools that can be called by VLMs during analysis,
including zoom, crop, contrast adjustment, and intensity thresholding.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from beartype import beartype
from PIL import Image

from nova_retrieval_vlm.visual_reasoning.image_ops import adjust_contrast
from nova_retrieval_vlm.visual_reasoning.image_ops import apply_intensity_threshold
from nova_retrieval_vlm.visual_reasoning.image_ops import crop_image
from nova_retrieval_vlm.visual_reasoning.image_ops import zoom_image


@dataclass
class ToolResult:
    """Result of executing a visual tool."""

    success: bool
    tool_name: str
    description: str
    image_base64: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class VisualTool:
    """A visual tool that can be called by the model."""

    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    execute: Callable[..., ToolResult]


def _image_to_base64(image: Image.Image, quality: int = 85) -> str:
    """Convert PIL Image to base64 JPEG string."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class ToolRegistry:
    """Registry of visual tools available for agentic analysis."""

    def __init__(self, image_path: Path | None = None):
        """Initialize tool registry with optional source image."""
        self._tools: dict[str, VisualTool] = {}
        self._image_path = image_path
        self._current_image: Image.Image | None = None
        self._tool_history: list[ToolResult] = []

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of visual tools."""
        self.register(
            VisualTool(
                name="zoom",
                description="Zoom into the image by a scale factor. Useful for examining fine details.",
                parameters={
                    "factor": {
                        "type": "number",
                        "description": "Zoom factor (1.0 = no change, 2.0 = 2x zoom)",
                        "minimum": 0.5,
                        "maximum": 4.0,
                    }
                },
                execute=self._execute_zoom,
            )
        )

        self.register(
            VisualTool(
                name="crop",
                description="Crop a region of interest from the image using normalized coordinates.",
                parameters={
                    "box": {
                        "type": "array",
                        "description": "Bounding box as [x1, y1, x2, y2] in normalized 0-1 coordinates",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 4,
                        "maxItems": 4,
                    }
                },
                execute=self._execute_crop,
            )
        )

        self.register(
            VisualTool(
                name="adjust_contrast",
                description="Adjust image contrast to enhance visibility of structures.",
                parameters={
                    "factor": {
                        "type": "number",
                        "description": "Contrast factor (1.0 = no change, >1.0 = more contrast)",
                        "minimum": 0.5,
                        "maximum": 3.0,
                    }
                },
                execute=self._execute_contrast,
            )
        )

        self.register(
            VisualTool(
                name="threshold",
                description="Apply intensity threshold to isolate specific intensity ranges (e.g., CSF, lesions).",
                parameters={
                    "lower": {
                        "type": "integer",
                        "description": "Lower intensity bound (0-255)",
                        "minimum": 0,
                        "maximum": 254,
                    },
                    "upper": {
                        "type": "integer",
                        "description": "Upper intensity bound (0-255)",
                        "minimum": 1,
                        "maximum": 255,
                    },
                },
                execute=self._execute_threshold,
            )
        )

        self.register(
            VisualTool(
                name="reset",
                description="Reset to the original image, discarding all modifications.",
                parameters={},
                execute=self._execute_reset,
            )
        )

    @beartype
    def register(self, tool: VisualTool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool

    @beartype
    def set_image(self, image_path: Path) -> None:
        """Set the source image for tool operations."""
        self._image_path = image_path
        self._current_image = Image.open(image_path)
        self._tool_history.clear()

    @beartype
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for all registered tools."""
        schemas = []
        for tool in self._tools.values():
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": list(tool.parameters.keys()),
                    },
                },
            }
            schemas.append(schema)
        return schemas

    @beartype
    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with given arguments."""
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                description=f"Unknown tool: {tool_name}",
                error=f"Tool '{tool_name}' not found in registry",
            )

        if self._current_image is None and self._image_path is not None:
            self._current_image = Image.open(self._image_path)

        if self._current_image is None:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                description="No image loaded",
                error="No image has been set for tool operations",
            )

        result = self._tools[tool_name].execute(**kwargs)
        self._tool_history.append(result)
        return result

    @property
    def history(self) -> list[ToolResult]:
        """Get the history of tool executions."""
        return self._tool_history.copy()

    @property
    def current_image(self) -> Image.Image | None:
        """Get the current working image."""
        return self._current_image

    def _execute_zoom(self, factor: float) -> ToolResult:
        """Execute zoom tool."""
        if self._current_image is None:
            return ToolResult(
                success=False,
                tool_name="zoom",
                description="No image loaded",
                error="No image available",
            )

        self._current_image = zoom_image(self._current_image, factor)
        return ToolResult(
            success=True,
            tool_name="zoom",
            description=f"Zoomed image by factor {factor:.1f}x",
            image_base64=_image_to_base64(self._current_image),
            metadata={"factor": factor, "new_size": self._current_image.size},
        )

    def _execute_crop(self, box: list[float]) -> ToolResult:
        """Execute crop tool."""
        if self._current_image is None:
            return ToolResult(
                success=False,
                tool_name="crop",
                description="No image loaded",
                error="No image available",
            )

        box_tuple = (box[0], box[1], box[2], box[3])
        self._current_image = crop_image(self._current_image, box_tuple)
        return ToolResult(
            success=True,
            tool_name="crop",
            description=f"Cropped region [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}]",
            image_base64=_image_to_base64(self._current_image),
            metadata={"box": box, "new_size": self._current_image.size},
        )

    def _execute_contrast(self, factor: float) -> ToolResult:
        """Execute contrast adjustment tool."""
        if self._current_image is None:
            return ToolResult(
                success=False,
                tool_name="adjust_contrast",
                description="No image loaded",
                error="No image available",
            )

        self._current_image = adjust_contrast(self._current_image, factor)
        return ToolResult(
            success=True,
            tool_name="adjust_contrast",
            description=f"Adjusted contrast by factor {factor:.1f}",
            image_base64=_image_to_base64(self._current_image),
            metadata={"factor": factor},
        )

    def _execute_threshold(self, lower: int, upper: int) -> ToolResult:
        """Execute intensity threshold tool."""
        if self._current_image is None:
            return ToolResult(
                success=False,
                tool_name="threshold",
                description="No image loaded",
                error="No image available",
            )

        self._current_image = apply_intensity_threshold(self._current_image, lower, upper)
        return ToolResult(
            success=True,
            tool_name="threshold",
            description=f"Applied threshold [{lower}, {upper}]",
            image_base64=_image_to_base64(self._current_image),
            metadata={"lower": lower, "upper": upper},
        )

    def _execute_reset(self) -> ToolResult:
        """Reset to original image."""
        if self._image_path is None:
            return ToolResult(
                success=False,
                tool_name="reset",
                description="No original image path",
                error="Cannot reset - no original image path stored",
            )

        self._current_image = Image.open(self._image_path)
        return ToolResult(
            success=True,
            tool_name="reset",
            description="Reset to original image",
            image_base64=_image_to_base64(self._current_image),
            metadata={"original_size": self._current_image.size},
        )
