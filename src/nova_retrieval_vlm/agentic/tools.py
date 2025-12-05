"""Visual tools for agentic medical image analysis.

Provides a registry of tools that can be called by VLMs during analysis,
including zoom, crop, contrast adjustment, intensity thresholding,
flipping, rotation, and web/image search.
"""

from __future__ import annotations

import asyncio
import base64
import io
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger
from PIL import Image

from nova_retrieval_vlm.retrieval.image_search import ImageSearchError
from nova_retrieval_vlm.retrieval.image_search import search_medical_images
from nova_retrieval_vlm.retrieval.web_search import SearchError
from nova_retrieval_vlm.retrieval.web_search import search_medical_literature
from nova_retrieval_vlm.visual_reasoning.image_ops import adjust_contrast
from nova_retrieval_vlm.visual_reasoning.image_ops import apply_intensity_threshold
from nova_retrieval_vlm.visual_reasoning.image_ops import crop_image
from nova_retrieval_vlm.visual_reasoning.image_ops import flip_horizontal
from nova_retrieval_vlm.visual_reasoning.image_ops import flip_vertical
from nova_retrieval_vlm.visual_reasoning.image_ops import rotate_90
from nova_retrieval_vlm.visual_reasoning.image_ops import zoom_image


class ToolExecutionError(Exception):
    """Raised when a tool execution fails due to invalid state."""


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
    execute: Callable[..., Awaitable[ToolResult]]
    requires_image: bool = True


@beartype
def image_to_base64(image: Image.Image, quality: int = 85) -> str:
    """Convert PIL Image to base64 JPEG string."""
    with io.BytesIO() as buffer:
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class ToolRegistry:
    """Registry of visual tools available for agentic analysis.

    Supports async context manager protocol for proper resource cleanup:
        async with ToolRegistry(image_path) as registry:
            result = await registry.execute("zoom", factor=2.0)
    """

    @beartype
    def __init__(
        self,
        image_path: Path | None = None,
        disabled_tools: list[str] | None = None,
    ) -> None:
        """Initialize tool registry with optional source image.

        Args:
            image_path: Path to the source image for tool operations
            disabled_tools: List of tool names to exclude from registration
        """
        self._tools: dict[str, VisualTool] = {}
        self._image_path = image_path
        self._current_image: Image.Image | None = None
        self._tool_history: list[ToolResult] = []
        self._image_lock = asyncio.Lock()
        self._disabled_tools = set(disabled_tools or [])
        self._register_default_tools()

    async def __aenter__(self) -> ToolRegistry:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit with cleanup."""
        self.close()

    @beartype
    def close(self) -> None:
        """Close and release all resources."""
        if self._current_image is not None:
            self._current_image.close()
            self._current_image = None
        self._tool_history.clear()

    @contextmanager
    def _update_image(self) -> Iterator[Image.Image]:
        """Context manager for safe image updates with automatic cleanup.

        Yields the current image, then replaces it with whatever was assigned
        to self._current_image and closes the old image.

        Usage:
            with self._update_image() as img:
                self._current_image = some_operation(img)
        """
        if self._current_image is None:
            raise ToolExecutionError("No image loaded")
        old_image = self._current_image
        try:
            yield old_image
        finally:
            if self._current_image is not old_image:
                old_image.close()

    def _should_register(self, tool_name: str) -> bool:
        """Check if a tool should be registered based on disabled_tools config."""
        return tool_name not in self._disabled_tools

    @beartype
    def _register_default_tools(self) -> None:
        """Register the default set of visual tools."""
        if self._should_register("zoom"):
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

        if self._should_register("crop"):
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

        if self._should_register("adjust_contrast"):
            self.register(
                VisualTool(
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
                    execute=self._execute_contrast,
                )
            )

        if self._should_register("threshold"):
            self.register(
                VisualTool(
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
                    execute=self._execute_threshold,
                )
            )

        if self._should_register("flip_horizontal"):
            self.register(
                VisualTool(
                    name="flip_horizontal",
                    description="Mirror the image left-right for bilateral symmetry assessment.",
                    parameters={},
                    execute=self._execute_flip_horizontal,
                )
            )

        if self._should_register("flip_vertical"):
            self.register(
                VisualTool(
                    name="flip_vertical",
                    description="Mirror the image top-bottom for orientation standardization.",
                    parameters={},
                    execute=self._execute_flip_vertical,
                )
            )

        if self._should_register("rotate"):
            self.register(
                VisualTool(
                    name="rotate",
                    description="Rotate image by 90 degrees clockwise or counter-clockwise.",
                    parameters={
                        "clockwise": {
                            "type": "boolean",
                            "description": "If true, rotate clockwise; if false, counter-clockwise.",
                            "default": True,
                        }
                    },
                    execute=self._execute_rotate,
                )
            )

        if self._should_register("reset"):
            self.register(
                VisualTool(
                    name="reset",
                    description="Return to the original full image, discarding all modifications.",
                    parameters={},
                    execute=self._execute_reset,
                )
            )

        if self._should_register("search_web"):
            self.register(
                VisualTool(
                    name="search_web",
                    description=(
                        "Search PubMed for medical literature, guidelines, "
                        "research papers, and reference cases."
                    ),
                    parameters={
                        "query": {
                            "type": "string",
                            "description": (
                                "Medical search query. Include condition, imaging "
                                "modality, and key findings."
                            ),
                        },
                        "search_type": {
                            "type": "string",
                            "description": "Type of search.",
                            "enum": ["diagnosis", "research", "guidelines", "anatomy", "general"],
                            "default": "general",
                        },
                    },
                    execute=self._execute_search_web,
                    requires_image=False,
                )
            )

        if self._should_register("search_images"):
            self.register(
                VisualTool(
                    name="search_images",
                    description=(
                        "Search NIH Open-i for reference medical images. "
                        "Returns image URLs with captions and metadata."
                    ),
                    parameters={
                        "query": {
                            "type": "string",
                            "description": "Medical image search query.",
                        },
                        "modality": {
                            "type": "string",
                            "description": "Filter by imaging modality.",
                            "enum": ["MRI", "CT", "X-ray", "Ultrasound", "PET", "Mammography", "any"],
                        },
                        "body_part": {
                            "type": "string",
                            "description": "Filter by body part.",
                            "enum": [
                                "brain",
                                "head",
                                "chest",
                                "abdomen",
                                "spine",
                                "pelvis",
                                "cardiac",
                                "any",
                            ],
                        },
                    },
                    execute=self._execute_search_images,
                    requires_image=False,
                )
            )

    @beartype
    def register(self, tool: VisualTool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool

    @beartype
    def set_image(self, image_path: Path) -> None:
        """Set the source image for tool operations."""
        if self._current_image is not None:
            self._current_image.close()
        self._image_path = image_path
        with Image.open(image_path) as img:
            self._current_image = img.copy()
        self._tool_history.clear()

    @beartype
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for all registered tools."""
        schemas = []
        for tool in self._tools.values():
            properties = {}
            required_params = []

            for param_name, param_def in tool.parameters.items():
                prop: dict[str, Any] = {"type": param_def.get("type", "string")}
                if "description" in param_def:
                    prop["description"] = param_def["description"]
                if "enum" in param_def:
                    prop["enum"] = param_def["enum"]
                if "default" in param_def:
                    prop["default"] = param_def["default"]
                else:
                    required_params.append(param_name)
                if param_def.get("type") == "array" and "items" in param_def:
                    prop["items"] = param_def["items"]
                properties[param_name] = prop

            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required_params,
                        "additionalProperties": False,
                    },
                },
            }
            schemas.append(schema)
        return schemas

    @beartype
    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with given arguments."""
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                description=f"Unknown tool: {tool_name}",
                error=f"Tool '{tool_name}' not found in registry",
            )

        tool = self._tools[tool_name]

        if tool.requires_image:
            await self._ensure_image_loaded()
            if self._current_image is None:
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    description="No image loaded",
                    error="No image has been set for tool operations",
                )

        result = await tool.execute(**kwargs)
        self._tool_history.append(result)
        return result

    @property
    def history(self) -> list[ToolResult]:
        """Get the history of tool executions."""
        return self._tool_history.copy()

    @beartype
    async def _ensure_image_loaded(self) -> None:
        """Lazy load the image with async lock to prevent race conditions."""
        if self._current_image is None and self._image_path is not None:
            async with self._image_lock:
                if self._current_image is None:
                    with Image.open(self._image_path) as img:
                        self._current_image = img.copy()

    @property
    def current_image(self) -> Image.Image | None:
        """Get the currently loaded image (may be None if not yet loaded)."""
        return self._current_image

    @beartype
    async def _execute_zoom(self, factor: float) -> ToolResult:
        """Execute zoom tool."""
        if not 0.5 <= factor <= 4.0:
            return ToolResult(
                success=False,
                tool_name="zoom",
                description=f"Invalid zoom factor: {factor:.2f}",
                error=f"Zoom factor must be between 0.5 and 4.0, got {factor:.2f}",
            )

        with self._update_image() as img:
            original_size = img.size
            self._current_image = zoom_image(img, factor)

        new_size = self._current_image.size
        return ToolResult(
            success=True,
            tool_name="zoom",
            description=f"Zoomed {factor:.1f}x: {original_size} -> {new_size} px",
            image_base64=image_to_base64(self._current_image),
            metadata={
                "factor": factor,
                "original_size": original_size,
                "new_size": new_size,
            },
        )

    @beartype
    async def _execute_crop(self, box: list[float]) -> ToolResult:
        """Execute crop tool."""
        if len(box) != 4:
            return ToolResult(
                success=False,
                tool_name="crop",
                description=f"Invalid crop box: expected 4 values, got {len(box)}",
                error=f"Crop requires [x1, y1, x2, y2], got {len(box)} values",
            )

        x1, y1, x2, y2 = box
        if not all(0 <= coord <= 1 for coord in box):
            return ToolResult(
                success=False,
                tool_name="crop",
                description=f"Crop coordinates out of range: {box}",
                error="All coordinates must be between 0 and 1",
            )

        if x2 <= x1 or y2 <= y1:
            return ToolResult(
                success=False,
                tool_name="crop",
                description=f"Invalid crop region: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]",
                error="Crop coordinates must satisfy x2 > x1 and y2 > y1",
            )

        with self._update_image() as img:
            original_size = img.size
            self._current_image = crop_image(img, (x1, y1, x2, y2))

        new_size = self._current_image.size
        area_percentage = (x2 - x1) * (y2 - y1) * 100

        return ToolResult(
            success=True,
            tool_name="crop",
            description=f"Cropped to [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}] ({area_percentage:.1f}%)",
            image_base64=image_to_base64(self._current_image),
            metadata={
                "box": box,
                "original_size": original_size,
                "new_size": new_size,
                "area_percentage": round(area_percentage, 1),
            },
        )

    @beartype
    async def _execute_contrast(self, factor: float) -> ToolResult:
        """Execute contrast adjustment tool."""
        if not 0.5 <= factor <= 3.0:
            return ToolResult(
                success=False,
                tool_name="adjust_contrast",
                description=f"Invalid contrast factor: {factor:.2f}",
                error=f"Contrast factor must be between 0.5 and 3.0, got {factor:.2f}",
            )

        with self._update_image() as img:
            original_size = img.size
            self._current_image = adjust_contrast(img, factor)

        return ToolResult(
            success=True,
            tool_name="adjust_contrast",
            description=f"Adjusted contrast by factor {factor:.1f}x",
            image_base64=image_to_base64(self._current_image),
            metadata={"factor": factor, "size": original_size},
        )

    @beartype
    async def _execute_threshold(self, lower: int, upper: int) -> ToolResult:
        """Execute intensity threshold tool."""
        if not 0 <= lower <= 254:
            return ToolResult(
                success=False,
                tool_name="threshold",
                description=f"Invalid lower bound: {lower}",
                error=f"Lower bound must be between 0 and 254, got {lower}",
            )

        if not 1 <= upper <= 255:
            return ToolResult(
                success=False,
                tool_name="threshold",
                description=f"Invalid upper bound: {upper}",
                error=f"Upper bound must be between 1 and 255, got {upper}",
            )

        if lower >= upper:
            return ToolResult(
                success=False,
                tool_name="threshold",
                description=f"Invalid range: [{lower}, {upper}]",
                error=f"Lower bound ({lower}) must be less than upper bound ({upper})",
            )

        with self._update_image() as img:
            original_size = img.size
            self._current_image = apply_intensity_threshold(img, lower, upper)

        return ToolResult(
            success=True,
            tool_name="threshold",
            description=f"Applied threshold [{lower}, {upper}]",
            image_base64=image_to_base64(self._current_image),
            metadata={"lower": lower, "upper": upper, "size": original_size},
        )

    @beartype
    async def _execute_flip_horizontal(self) -> ToolResult:
        """Execute horizontal flip tool."""
        with self._update_image() as img:
            self._current_image = flip_horizontal(img)

        return ToolResult(
            success=True,
            tool_name="flip_horizontal",
            description="Flipped image horizontally",
            image_base64=image_to_base64(self._current_image),
            metadata={"size": self._current_image.size},
        )

    @beartype
    async def _execute_flip_vertical(self) -> ToolResult:
        """Execute vertical flip tool."""
        with self._update_image() as img:
            self._current_image = flip_vertical(img)

        return ToolResult(
            success=True,
            tool_name="flip_vertical",
            description="Flipped image vertically",
            image_base64=image_to_base64(self._current_image),
            metadata={"size": self._current_image.size},
        )

    @beartype
    async def _execute_rotate(self, clockwise: bool = True) -> ToolResult:
        """Execute rotation tool."""
        with self._update_image() as img:
            original_size = img.size
            self._current_image = rotate_90(img, clockwise=clockwise)

        direction = "clockwise" if clockwise else "counter-clockwise"
        return ToolResult(
            success=True,
            tool_name="rotate",
            description=f"Rotated 90° {direction}",
            image_base64=image_to_base64(self._current_image),
            metadata={
                "clockwise": clockwise,
                "original_size": original_size,
                "new_size": self._current_image.size,
            },
        )

    @beartype
    async def _execute_reset(self) -> ToolResult:
        """Reset to original image."""
        if self._image_path is None:
            return ToolResult(
                success=False,
                tool_name="reset",
                description="No original image path",
                error="Cannot reset - no original image path stored",
            )

        if self._current_image is not None:
            self._current_image.close()
        with Image.open(self._image_path) as img:
            self._current_image = img.copy()

        return ToolResult(
            success=True,
            tool_name="reset",
            description="Reset to original image",
            image_base64=image_to_base64(self._current_image),
            metadata={"size": self._current_image.size},
        )

    @beartype
    async def _execute_search_web(self, query: str, search_type: str = "general") -> ToolResult:
        """Search PubMed for medical literature."""
        logger.info(f"Searching PubMed: '{query}' (type: {search_type})")

        try:
            search_results = await search_medical_literature(
                query=query,
                max_results=5,
                search_type=search_type,
            )
        except SearchError as e:
            return ToolResult(
                success=False,
                tool_name="search_web",
                description=f"PubMed search failed for '{query}'",
                error=str(e),
                metadata={"query": query, "search_type": search_type},
            )

        if not search_results:
            return ToolResult(
                success=True,
                tool_name="search_web",
                description=f"No results found for '{query}'",
                metadata={"query": query, "search_type": search_type, "results_count": 0},
            )

        formatted_results = []
        sources = []
        total_reliability = 0.0
        content_types = []

        for i, result in enumerate(search_results, 1):
            sources.append(result.source)
            total_reliability += result.reliability_score
            content_types.append(result.content_type)

            lines = [
                f"{i}. **{result.title}**",
                f"   **Source:** {result.source} (Reliability: {result.reliability_score:.2f})",
                f"   **Type:** {result.content_type} | **Open Access:** {'Yes' if result.open_access else 'No'}",
            ]
            if result.publication_date:
                lines.append(f"   **Date:** {result.publication_date}")
            if result.journal:
                lines.append(f"   **Journal:** {result.journal}")
            if result.content and result.content != result.title:
                content_preview = result.content[:500] + ("..." if len(result.content) > 500 else "")
                lines.append(f"   **Content:** {content_preview}")
            if result.extracted_entities:
                lines.append(f"   **Key terms:** {', '.join(result.extracted_entities[:5])}")
            lines.append(f"   **URL:** {result.url}")
            formatted_results.append("\n".join(lines))

        formatted_summary = "\n## PubMed Search Results\n\n" + "\n\n".join(formatted_results)
        avg_reliability = total_reliability / len(search_results)

        return ToolResult(
            success=True,
            tool_name="search_web",
            description=f"Found {len(search_results)} PubMed articles",
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

    @beartype
    async def _execute_search_images(
        self, query: str, modality: str = "any", body_part: str = "any"
    ) -> ToolResult:
        """Search NIH Open-i for reference medical images."""
        modality_filter = None if modality == "any" else modality
        body_part_filter = None if body_part == "any" else body_part
        logger.info(f"Searching images: '{query}' (modality: {modality_filter}, body: {body_part_filter})")

        try:
            search_results = await search_medical_images(
                query=query,
                max_results=5,
                modality=modality_filter,
                body_part=body_part_filter,
            )
        except ImageSearchError as e:
            return ToolResult(
                success=False,
                tool_name="search_images",
                description=f"Image search failed for '{query}'",
                error=str(e),
                metadata={"query": query, "modality": modality_filter, "body_part": body_part_filter},
            )

        if not search_results:
            return ToolResult(
                success=True,
                tool_name="search_images",
                description=f"No images found for '{query}'",
                metadata={
                    "query": query,
                    "modality": modality_filter,
                    "body_part": body_part_filter,
                    "results_count": 0,
                },
            )

        formatted_results = []
        modalities_found = []
        body_parts_found = []

        for i, result in enumerate(search_results, 1):
            if result.modality:
                modalities_found.append(result.modality)
            if result.body_part:
                body_parts_found.append(result.body_part)

            lines = [
                f"{i}. **{result.title}**",
                f"   **Source:** {result.source} (Reliability: {result.reliability_score:.2f})",
                f"   **Modality:** {result.modality or 'Unknown'} | **Body Part:** {result.body_part or 'Unknown'}",
            ]
            if result.caption:
                caption_preview = result.caption[:400] + ("..." if len(result.caption) > 400 else "")
                lines.append(f"   **Caption:** {caption_preview}")
            if result.article_title:
                lines.append(f"   **Article:** {result.article_title[:100]}")
            lines.append(f"   **Image URL:** {result.image_url}")
            lines.append(f"   **Source Article:** {result.source_url}")
            formatted_results.append("\n".join(lines))

        formatted_summary = "\n## Reference Medical Images\n\n" + "\n\n".join(formatted_results)

        return ToolResult(
            success=True,
            tool_name="search_images",
            description=f"Found {len(search_results)} reference images",
            metadata={
                "query": query,
                "modality": modality_filter,
                "body_part": body_part_filter,
                "results_count": len(search_results),
                "modalities_found": list(set(modalities_found)),
                "body_parts_found": list(set(body_parts_found)),
                "image_urls": [r.image_url for r in search_results],
                "formatted_results": formatted_summary,
            },
        )
