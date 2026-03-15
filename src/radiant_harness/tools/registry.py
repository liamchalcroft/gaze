"""Tool registry with image management, execution tracking, and documentation.

This module provides:
- ToolDocumenter: Schema generation and documentation formatting
- ToolRegistry: Complete tool management with image handling and execution
- EncodedImage: Container for base64-encoded image data
- encode_image: Utility to encode PIL Images
"""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING
from typing import Any

from beartype import beartype
from beartype.roar import BeartypeException
from PIL import Image

from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.exceptions import UnknownToolError
from radiant_harness.tools.image_manager import ImageManager
from radiant_harness.tools.tool import Tool
from radiant_harness.types import ToolResult

if TYPE_CHECKING:
    from radiant_harness.retrieval.image_search import MedicalImageSearchManager
    from radiant_harness.retrieval.web_search import WebSearchManager

# Valid JSON Schema types for tool parameters
_VALID_PARAM_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}


@dataclass(frozen=True)
class EncodedImage:
    """Container for encoded image data."""

    data: str
    mime_type: str
    _data_url: str = ""

    def __post_init__(self) -> None:
        # Pre-compute the data URL once to avoid re-creating the ~500KB+
        # string on every call.  Uses object.__setattr__ because the
        # dataclass is frozen.
        object.__setattr__(self, "_data_url", f"data:{self.mime_type};base64,{self.data}")

    def to_data_url(self) -> str:
        """Convert to a data URL for embedding in HTML/messages."""
        return self._data_url


@beartype
def encode_image(
    image: Image.Image,
    *,
    format: str = "JPEG",
    quality: int | None = None,
) -> EncodedImage:
    """Encode a PIL Image to a base64 string.

    Args:
        image: PIL Image to encode.
        format: Image format — ``"JPEG"`` (default, much smaller) or ``"PNG"``.
        quality: JPEG quality 1-100. Ignored for PNG. When *None*, uses
            ``ImageProcessingConfig.default_jpeg_quality`` (default 85).

    Returns:
        EncodedImage with base64 data and correct MIME type.
    """
    from radiant_harness.config import get_config

    fmt = format.upper()
    if fmt not in {"JPEG", "PNG"}:
        raise ValueError(f"Unsupported image format: {format!r}. Use 'JPEG' or 'PNG'.")

    # JPEG only supports RGB and L modes.  Medical images may use I (32-bit
    # int), I;16 (16-bit int from DICOM-converted PNGs), or F (float32).
    # Alpha modes (RGBA, LA, PA) and palette mode (P) also need conversion.
    jpeg_safe_modes = {"RGB", "L"}
    if fmt == "JPEG" and image.mode not in jpeg_safe_modes:
        image = image.convert("RGB")

    # PNG cannot save mode F (float32).  Convert to L for lossless grayscale.
    png_unsafe_modes = {"F"}
    if fmt == "PNG" and image.mode in png_unsafe_modes:
        image = image.convert("L")

    buffer = BytesIO()
    if fmt == "PNG":
        image.save(buffer, format="PNG")
        mime = "image/png"
    else:
        q = quality if quality is not None else get_config().image.default_jpeg_quality
        image.save(buffer, format="JPEG", quality=q)
        mime = "image/jpeg"

    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return EncodedImage(data=image_base64, mime_type=mime)


@beartype
class ToolDocumenter:
    """Generates tool schemas and documentation.

    Handles:
    - OpenAI-compatible tool schema generation
    - Prompt documentation formatting
    - Tool categorization and filtering
    - Schema validation

    Example:
        documenter = ToolDocumenter(tools=[zoom_tool, crop_tool])
        schemas = documenter.get_tool_schemas()
        docs = documenter.generate_prompt_documentation()
    """

    @beartype
    def __init__(self, tools: list[Tool] | None = None) -> None:
        """Initialize tool documenter.

        Args:
            tools: List of tools to document. Can be empty and tools added later.
        """
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    @beartype
    def register(self, tool: Tool) -> None:
        """Register a tool for documentation.

        Args:
            tool: Tool to register
        """
        self._tools[tool.name] = tool

    @beartype
    def get_tool(self, name: str) -> Tool | None:
        """Get a registered tool by name.

        Args:
            name: Tool name to look up

        Returns:
            Tool if found, None otherwise
        """
        return self._tools.get(name)

    @beartype
    def get_tool_names(self) -> list[str]:
        """Get list of all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    @beartype
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for all registered tools.

        Returns:
            List of tool schemas in OpenAI function-calling format

        Raises:
            ValueError: If tool has invalid schema configuration
        """
        schemas: list[dict[str, Any]] = []
        for tool in self._tools.values():
            properties: dict[str, Any] = {}
            required_params: list[str] = []

            for param_name, param_def in tool.parameters.items():
                # Validate parameter type
                param_type = param_def.get("type")
                if not param_type:
                    raise ValueError(
                        f"Tool '{tool.name}' parameter '{param_name}' is missing required 'type'"
                    )
                if param_type not in _VALID_PARAM_TYPES:
                    raise ValueError(
                        f"Tool '{tool.name}' parameter "
                        f"'{param_name}' has invalid type "
                        f"'{param_type}'. Must be one of: "
                        f"{', '.join(sorted(_VALID_PARAM_TYPES))}"
                    )

                prop: dict[str, Any] = {"type": param_type}

                # Copy schema validation keywords
                for key in (
                    "description",
                    "enum",
                    "default",
                    "minimum",
                    "maximum",
                    "minItems",
                    "maxItems",
                    "pattern",
                    "format",
                ):
                    if key in param_def:
                        value = param_def[key]
                        # Skip "default": None — emitting {"type": "integer", "default": null}
                        # is invalid JSON Schema.  The parameter is already marked optional
                        # (not in required) via the "default" key presence check below.
                        if key == "default" and value is None:
                            continue
                        prop[key] = value

                # Handle array item types
                if param_def.get("type") == "array" and "items" in param_def:
                    prop["items"] = param_def["items"]

                # Mark as required if no default
                if "default" not in param_def:
                    required_params.append(param_name)

                properties[param_name] = prop

            schema: dict[str, Any] = {
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
    def get_tools_by_category(self) -> dict[str, list[Tool]]:
        """Group registered tools by category.

        Returns:
            Dictionary mapping category names to lists of tools.
            Tools with no category are grouped under "other".
        """
        categories: dict[str, list[Tool]] = {}
        for tool in self._tools.values():
            category = tool.category or "other"
            if category not in categories:
                categories[category] = []
            categories[category].append(tool)
        return categories

    @beartype
    def generate_prompt_documentation(
        self,
        group_by_category: bool = True,
        include_categories: set[str] | None = None,
        exclude_categories: set[str] | None = None,
        compact: bool = False,
    ) -> str:
        """Generate prompt documentation for all registered tools.

        This creates formatted text suitable for inclusion in system prompts,
        documenting all available tools with their parameters and usage.

        Args:
            group_by_category: If True, group tools by category with headers
            include_categories: If set, only include tools from these categories
            exclude_categories: If set, exclude tools from these categories
            compact: If True, emit one-line-per-tool summaries to reduce token
                overhead for small-context models (<=8K).

        Returns:
            Formatted documentation string for system prompts
        """
        if not self._tools:
            return ""

        sections: list[str] = []

        if group_by_category:
            categories = self.get_tools_by_category()

            # Apply category filters
            if include_categories:
                categories = {k: v for k, v in categories.items() if k in include_categories}
            if exclude_categories:
                categories = {k: v for k, v in categories.items() if k not in exclude_categories}

            # Sort categories for consistent output
            for category in sorted(categories.keys()):
                tools = categories[category]
                if not tools:
                    continue

                # Category header
                category_title = category.replace("_", " ").title()
                sections.append(
                    f"**{category_title} Tools:**\n" if not compact else f"[{category_title}]"
                )

                # Tool documentation
                for tool in sorted(tools, key=lambda t: t.name):
                    sections.append(tool.get_prompt_documentation(compact=compact))
                    if not compact:
                        sections.append("")  # Blank line between tools
        else:
            # Flat list without categories
            tools = list(self._tools.values())

            # Apply category filters
            if include_categories:
                tools = [t for t in tools if (t.category or "other") in include_categories]
            if exclude_categories:
                tools = [t for t in tools if (t.category or "other") not in exclude_categories]

            for tool in sorted(tools, key=lambda t: t.name):
                sections.append(tool.get_prompt_documentation(compact=compact))
                if not compact:
                    sections.append("")

        return "\n".join(sections).strip()


class ToolRegistry:
    """Refactored tool registry with separated responsibilities.

    This implementation delegates to specialized managers:
    - ImageManager: Handles image loading and transformation
    - ToolDocumenter: Handles schema generation and documentation

    Architecture:
        ToolRegistry
        ├── ImageManager (image loading, transformation, state)
        ├── ToolDocumenter (schemas, documentation, validation)
        └── Tool execution (actual tool calling)
    """

    @beartype
    def __init__(
        self,
        image_path: Path | None = None,
        tools: list[Tool] | None = None,
        max_history: int = 100,
        web_search_manager: Any | None = None,
        image_search_manager: Any | None = None,
    ) -> None:
        """Initialize refactored tool registry."""
        # Initialize specialized managers
        self._image_manager = ImageManager()
        self._documenter = ToolDocumenter(tools)

        # Tool execution history
        self._tool_history: deque[ToolResult] = deque(maxlen=max_history)
        self.max_history = max_history

        # Lazily-created search managers — reused across tool calls within a
        # single agentic session to keep TCP connections alive.
        self._web_search_manager = web_search_manager
        self._image_search_manager = image_search_manager
        self._owns_web_search_manager = web_search_manager is None
        self._owns_image_search_manager = image_search_manager is None

        # Set initial image if provided
        if image_path:
            self._image_manager.set_image(image_path)

    def __enter__(self) -> ToolRegistry:
        """Sync context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Sync context manager exit with cleanup."""
        self.close()

    @beartype
    def close(self) -> None:
        """Close and clean up synchronous resources.

        Prefer :meth:`aclose` from async contexts so that search manager
        sessions are properly awaited.
        """
        self._image_manager.close()
        self._tool_history.clear()

    async def aclose(self) -> None:
        """Close all resources including async search manager sessions."""
        close_tasks: list[asyncio.Task[None]] = []
        if self._web_search_manager is not None and self._owns_web_search_manager:
            close_tasks.append(asyncio.ensure_future(self._web_search_manager.close()))
        if self._image_search_manager is not None and self._owns_image_search_manager:
            close_tasks.append(asyncio.ensure_future(self._image_search_manager.close()))
        self._web_search_manager = None
        self._image_search_manager = None
        if close_tasks:
            await asyncio.gather(*close_tasks)
        self.close()

    def get_web_search_manager(self) -> WebSearchManager:
        """Get or create a reusable WebSearchManager for the session."""
        if self._web_search_manager is None:
            from radiant_harness.retrieval.web_search import WebSearchManager

            self._web_search_manager = WebSearchManager()
        return self._web_search_manager

    def get_image_search_manager(self) -> MedicalImageSearchManager:
        """Get or create a reusable MedicalImageSearchManager for the session."""
        if self._image_search_manager is None:
            from radiant_harness.retrieval.image_search import MedicalImageSearchManager

            self._image_search_manager = MedicalImageSearchManager()
        return self._image_search_manager

    @beartype
    def register(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self._documenter.register(tool)

    @beartype
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for all registered tools."""
        return self._documenter.get_tool_schemas()

    @beartype
    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with given arguments.

        Raises:
            UnknownToolError: If ``tool_name`` is not registered.
            ToolExecutionError: If the tool raises a ``ToolExecutionError`` or
                if the call fails due to invalid argument types (``TypeError``).
        """
        tool = self._documenter.get_tool(tool_name)
        if tool is None:
            raise UnknownToolError(tool_name, self._documenter.get_tool_names())

        if tool.requires_image:
            await self._image_manager.ensure_loaded()
            if not self._image_manager.has_image:
                raise ToolExecutionError(
                    f"Tool '{tool_name}' requires an image, but no image path was provided"
                )

        # Coerce JSON-native types to match Python type hints that beartype enforces.
        # JSON deserializes 2 as int, but beartype rejects int for float hints;
        # similarly some models send 50.0 for integer-declared params.
        for param_name, param_info in tool.parameters.items():
            if param_name not in kwargs:
                continue
            val = kwargs[param_name]
            param_type = param_info.get("type")

            # int → float for "number" params
            if param_type == "number":
                if isinstance(val, int) and not isinstance(val, bool):
                    kwargs[param_name] = float(val)

            # float → int for "integer" params (only if lossless)
            elif param_type == "integer":
                if isinstance(val, float) and not isinstance(val, bool) and val == int(val):
                    kwargs[param_name] = int(val)

            # Coerce elements inside "array" params whose items are "number" or "integer"
            elif param_type == "array" and isinstance(val, list):
                items_type = param_info.get("items", {}).get("type")
                if items_type == "number":
                    kwargs[param_name] = [
                        float(v) if isinstance(v, int) and not isinstance(v, bool) else v
                        for v in val
                    ]
                elif items_type == "integer":
                    kwargs[param_name] = [
                        int(v)
                        if isinstance(v, float) and not isinstance(v, bool) and v == int(v)
                        else v
                        for v in val
                    ]

        try:
            result = await tool.execute(self, **kwargs)
        except (TypeError, BeartypeException) as e:
            raise ToolExecutionError(f"Tool '{tool_name}' received invalid arguments: {e}") from e

        self._tool_history.append(result)
        return result

    @property
    def history(self) -> list[ToolResult]:
        """Get the history of tool executions."""
        return list(self._tool_history)

    @beartype
    def get_image_manager(self) -> ImageManager:
        """Get the image manager instance."""
        return self._image_manager

    @beartype
    def get_documenter(self) -> ToolDocumenter:
        """Get the tool documenter instance."""
        return self._documenter


# Export classes for direct use
__all__ = [
    "ToolRegistry",
    "Tool",
    "EncodedImage",
    "encode_image",
    "ImageManager",
    "ToolDocumenter",
]
