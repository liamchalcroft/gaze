"""Visual tools for agentic medical image analysis.

Provides a registry of tools that can be called by VLMs during analysis,
including zoom, crop, contrast adjustment, intensity thresholding,
flipping, rotation, and visual retrieval (planned).
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
from loguru import logger
from PIL import Image

from nova_retrieval_vlm.retrieval.web_search import search_medical_literature_sync
from nova_retrieval_vlm.visual_reasoning.image_ops import adjust_contrast
from nova_retrieval_vlm.visual_reasoning.image_ops import apply_intensity_threshold
from nova_retrieval_vlm.visual_reasoning.image_ops import crop_image
from nova_retrieval_vlm.visual_reasoning.image_ops import flip_horizontal
from nova_retrieval_vlm.visual_reasoning.image_ops import flip_vertical
from nova_retrieval_vlm.visual_reasoning.image_ops import rotate_90
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

        # Note: Web search manager initialized lazily to avoid async/sync issues
        self._web_searcher = None

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of visual tools."""
        # Zoom tool
        self.register(
            VisualTool(
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
                execute=self._execute_zoom,
            )
        )

        # Crop tool
        self.register(
            VisualTool(
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
                execute=self._execute_crop,
            )
        )

        # Contrast adjustment tool
        self.register(
            VisualTool(
                name="adjust_contrast",
                description="Enhance or reduce image contrast to better distinguish tissue boundaries and subtle findings. Critical for detecting low-contrast lesions.",
                parameters={
                    "factor": {
                        "type": "number",
                        "description": "Contrast factor: 1.0=no change, >1.0=increase contrast, <1.0=decrease contrast. Try 1.5-2.5 for subtle findings.",
                        "minimum": 0.5,
                        "maximum": 3.0,
                    }
                },
                execute=self._execute_contrast,
            )
        )

        # Intensity threshold tool
        self.register(
            VisualTool(
                name="threshold",
                description="Apply intensity windowing to highlight specific tissue types or abnormalities. Essential for MRI density analysis.",
                parameters={
                    "lower": {
                        "type": "integer",
                        "description": "Lower bound: Minimum intensity to display (0-255). Use 50-120 for dark tissues, 150-200 for bright tissues.",
                        "minimum": 0,
                        "maximum": 254,
                    },
                    "upper": {
                        "type": "integer",
                        "description": "Upper bound: Maximum intensity to display (0-255). Must be higher than lower bound.",
                        "minimum": 1,
                        "maximum": 255,
                    },
                },
                execute=self._execute_threshold,
            )
        )

        # Flip horizontal tool
        self.register(
            VisualTool(
                name="flip_horizontal",
                description="Mirror the image left-right. Critical for assessing bilateral symmetry and comparing hemispheric structures.",
                parameters={},
                execute=self._execute_flip_horizontal,
            )
        )

        # Flip vertical tool
        self.register(
            VisualTool(
                name="flip_vertical",
                description="Mirror the image top-bottom. Useful for standardizing orientation and viewing from different perspectives.",
                parameters={},
                execute=self._execute_flip_vertical,
            )
        )

        # Rotate tool
        self.register(
            VisualTool(
                name="rotate",
                description="Rotate image by 90 degrees clockwise or counterclockwise. Essential for standardizing view orientation and examining anatomical relationships from different angles.",
                parameters={
                    "clockwise": {
                        "type": "boolean",
                        "description": "If true, rotate clockwise; if false, rotate counter-clockwise",
                        "default": True,
                    }
                },
                execute=self._execute_rotate,
            )
        )

        # Reset tool
        self.register(
            VisualTool(
                name="reset",
                description="Return to the original full image, discarding all previous modifications. Essential when you need to start over or examine the complete context after focused analysis.",
                parameters={},
                execute=self._execute_reset,
            )
        )

        # Web search tool for medical information
        self.register(
            VisualTool(
                name="search_web",
                description="Search the web for current medical information, guidelines, research papers, and reference cases. Use Radiopaedia, PubMed, and medical sites to verify findings and explore differential diagnoses.",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "Medical search query. Include condition, imaging modality, and key findings. Examples: 'glioblastoma MRI T1 contrast enhancement', 'cerebellar hemangioblastoma radiopaedia', 'brain metastasis differential diagnosis'",
                    },
                    "search_type": {
                        "type": "string",
                        "description": "Type of search: 'diagnosis' for clinical info, 'research' for recent studies, 'guidelines' for protocols, 'anatomy' for normal variants, or 'general' for broad search. Default: 'general'.",
                        "enum": ["diagnosis", "research", "guidelines", "anatomy", "general"],
                        "default": "general",
                    },
                },
                execute=self._execute_search_web,
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
        """Get the current working image (lazy loads if needed)."""
        if self._current_image is None and self._image_path is not None:
            self._current_image = Image.open(self._image_path)
        return self._current_image

    def _execute_zoom(self, factor: float) -> ToolResult:
        """Execute zoom tool."""
        if self._current_image is None:
            return ToolResult(
                success=False,
                tool_name="zoom",
                description="No image loaded for zoom operation",
                error="Cannot zoom - no image has been loaded. Please ensure an image is set before using tools.",
            )

        if factor < 0.5 or factor > 4.0:
            return ToolResult(
                success=False,
                tool_name="zoom",
                description=f"Invalid zoom factor: {factor:.2f}",
                error=f"Zoom factor {factor:.2f} outside valid range [0.5, 4.0]. "
                f"Recommend using 2.0-3.0 for detailed examination, or 0.5-1.0 for overview.",
            )

        original_size = self._current_image.size
        self._current_image = zoom_image(self._current_image, factor)
        new_size = self._current_image.size

        guidance = ""
        if factor > 2.5:
            guidance = " Warning: High zoom factor may reduce image quality. Consider using crop tool for focused examination."
        elif factor < 1.0:
            guidance = " Use this for an overview before detailed analysis."

        return ToolResult(
            success=True,
            tool_name="zoom",
            description=f"Zoomed image by factor {factor:.1f}x from {original_size} to {new_size} pixels.{guidance}",
            image_base64=_image_to_base64(self._current_image),
            metadata={
                "factor": factor,
                "original_size": original_size,
                "new_size": new_size,
                "size_change": f"{new_size[0] / original_size[0]:.2f}x",
            },
        )

    def _execute_crop(self, box: list[float]) -> ToolResult:
        """Execute crop tool."""
        if self._current_image is None:
            return ToolResult(
                success=False,
                tool_name="crop",
                description="No image loaded for crop operation",
                error="Cannot crop - no image has been loaded. Please ensure an image is set before using tools.",
            )

        if len(box) != 4:
            return ToolResult(
                success=False,
                tool_name="crop",
                description=f"Invalid crop box format: {box}",
                error=f"Crop requires 4 coordinates [x1, y1, x2, y2], got {len(box)} values. "
                f"Coordinates should be normalized (0-1). Example: [0.2, 0.3, 0.8, 0.7] for center region.",
            )

        x1, y1, x2, y2 = box
        if not all(0 <= coord <= 1 for coord in box):
            return ToolResult(
                success=False,
                tool_name="crop",
                description=f"Crop coordinates out of range: {box}",
                error="All crop coordinates must be between 0 and 1 (normalized). "
                "Got values outside this range. Please check your coordinate selection.",
            )

        if x2 <= x1 or y2 <= y1:
            return ToolResult(
                success=False,
                tool_name="crop",
                description=f"Invalid crop region: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]",
                error="Crop coordinates must satisfy x2 > x1 and y2 > y1. "
                "Your selection creates an invalid rectangle. Check your coordinate order.",
            )

        original_size = self._current_image.size
        box_tuple = (x1, y1, x2, y2)
        self._current_image = crop_image(self._current_image, box_tuple)
        new_size = self._current_image.size

        area_percentage = (x2 - x1) * (y2 - y1) * 100
        guidance = ""
        if area_percentage < 5:
            guidance = " Small crop area selected. Ensure this contains the region of interest."
        elif area_percentage > 80:
            guidance = (
                " Large crop area selected. Consider if more focused cropping would be beneficial."
            )

        return ToolResult(
            success=True,
            tool_name="crop",
            description=f"Cropped to region [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}] ({area_percentage:.1f}% of image, {new_size} pixels).{guidance}",
            image_base64=_image_to_base64(self._current_image),
            metadata={
                "box": box,
                "original_size": original_size,
                "new_size": new_size,
                "area_percentage": round(area_percentage, 1),
                "pixel_box": [
                    int(x1 * original_size[0]),
                    int(y1 * original_size[1]),
                    int(x2 * original_size[0]),
                    int(y2 * original_size[1]),
                ],
            },
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

    def _execute_flip_horizontal(self) -> ToolResult:
        """Execute horizontal flip tool."""
        if self._current_image is None:
            return ToolResult(
                success=False,
                tool_name="flip_horizontal",
                description="No image loaded",
                error="No image available",
            )

        self._current_image = flip_horizontal(self._current_image)
        return ToolResult(
            success=True,
            tool_name="flip_horizontal",
            description="Flipped image horizontally (left-right)",
            image_base64=_image_to_base64(self._current_image),
            metadata={"operation": "flip_horizontal"},
        )

    def _execute_flip_vertical(self) -> ToolResult:
        """Execute vertical flip tool."""
        if self._current_image is None:
            return ToolResult(
                success=False,
                tool_name="flip_vertical",
                description="No image loaded",
                error="No image available",
            )

        self._current_image = flip_vertical(self._current_image)
        return ToolResult(
            success=True,
            tool_name="flip_vertical",
            description="Flipped image vertically (top-bottom)",
            image_base64=_image_to_base64(self._current_image),
            metadata={"operation": "flip_vertical"},
        )

    def _execute_rotate(self, clockwise: bool = True) -> ToolResult:
        """Execute rotation tool."""
        if self._current_image is None:
            return ToolResult(
                success=False,
                tool_name="rotate",
                description="No image loaded",
                error="No image available",
            )

        self._current_image = rotate_90(self._current_image, clockwise=clockwise)
        direction = "clockwise" if clockwise else "counter-clockwise"
        return ToolResult(
            success=True,
            tool_name="rotate",
            description=f"Rotated image 90 degrees {direction}",
            image_base64=_image_to_base64(self._current_image),
            metadata={"clockwise": clockwise, "new_size": self._current_image.size},
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

    def _execute_search_web(self, query: str, search_type: str = "general") -> ToolResult:
        """Search PubMed for medical literature with enhanced reliability scoring."""
        try:
            logger.info(f"Searching PubMed for: '{query}' (type: {search_type})")

            # Perform synchronous medical literature search
            search_results = search_medical_literature_sync(
                query=query,
                max_results=5,
                search_type=search_type,
            )

            if not search_results:
                return ToolResult(
                    success=False,
                    tool_name="search_web",
                    description=f"No PubMed results found for '{query}'",
                    error="No relevant medical literature found in PubMed",
                    metadata={"query": query, "search_type": search_type, "results_count": 0},
                )

            # Format results for LLM consumption
            formatted_results = []
            sources = []
            avg_reliability = 0.0
            content_types = []

            for i, result in enumerate(search_results, 1):
                sources.append(result.source)
                avg_reliability += result.reliability_score
                content_types.append(result.content_type)

                # Create detailed result summary
                summary = f"{i}. **{result.title}**\n"
                summary += f"   **Source:** {result.source} (Reliability: {result.reliability_score:.2f})\n"
                summary += f"   **Type:** {result.content_type} | **Open Access:** {'Yes' if result.open_access else 'No'}\n"
                if result.publication_date:
                    summary += f"   **Date:** {result.publication_date}\n"
                if result.journal:
                    summary += f"   **Journal:** {result.journal}\n"

                # Add content/abstract
                if result.content and result.content != result.title:
                    summary += f"   **Content:** {result.content[:500]}{'...' if len(result.content) > 500 else ''}\n"
                else:
                    summary += "   **Summary:** No abstract available\n"

                # Add key entities if found
                if hasattr(result, "extracted_entities") and result.extracted_entities:
                    summary += f"   **Key terms:** {', '.join(result.extracted_entities[:5])}\n"

                summary += f"   **URL:** {result.url}\n"
                formatted_results.append(summary)

            # Calculate average reliability
            avg_reliability = avg_reliability / len(search_results)

            # Combine all results
            formatted_summary = "\n## PubMed Search Results\n\n" + "\n\n".join(formatted_results)

            return ToolResult(
                success=True,
                tool_name="search_web",
                description=f"Found {len(search_results)} PubMed articles",
                image_base64=None,
                metadata={
                    "query": query,
                    "search_type": search_type,
                    "results_count": len(search_results),
                    "sources": list(set(sources)),
                    "avg_reliability": round(avg_reliability, 2),
                    "content_types": list(set(content_types)),
                    "open_access_count": sum(1 for r in search_results if r.open_access),
                    "formatted_results": formatted_summary,
                },
            )

        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return ToolResult(
                success=False,
                tool_name="search_web",
                description=f"PubMed search failed for '{query}'",
                error=f"PubMed search error: {str(e)}",
                metadata={"query": query, "search_type": search_type, "error": str(e)},
            )
