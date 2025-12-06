"""Generic tool registry for the radiology VLM agent harness.

Provides a registry for visual and search tools that can be called
by VLMs during multi-turn analysis.
"""

from __future__ import annotations

import asyncio
import base64
import io
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beartype import beartype
from PIL import Image

from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.exceptions import UnknownToolError
from radiant_harness.types import ToolResult


@dataclass
class Tool:
    """A tool that can be called by the model.

    Attributes:
        name: Unique identifier for this tool
        description: Short description for the API schema (1-2 sentences)
        parameters: JSON Schema-style parameter definitions
        execute: Async function to execute the tool
        requires_image: Whether this tool requires an image to be loaded
        prompt_documentation: Detailed documentation for system prompts.
            This should include usage examples, parameter details, and
            guidance on when/how to use the tool. If None, uses description.
        category: Optional category for grouping tools (e.g., "visual", "search")

    Example:
        Tool(
            name="zoom",
            description="Magnify the image for detailed examination",
            parameters={"factor": {"type": "number", "description": "Zoom factor"}},
            execute=_execute_zoom,
            prompt_documentation='''
            **zoom** - Magnify the image for detailed examination
              - Parameter `factor` (number, 0.5-4.0): Magnification level
              - Use for: Examining small lesions, tissue boundaries, subtle findings
              - Example: zoom(factor=2.0) for 2x magnification
            ''',
            category="visual",
        )
    """

    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    execute: Callable[..., Awaitable[ToolResult]]
    requires_image: bool = True
    prompt_documentation: str | None = None
    category: str | None = None

    def get_prompt_documentation(self) -> str:
        """Get the prompt documentation, falling back to description if not set."""
        if self.prompt_documentation:
            return self.prompt_documentation.strip()
        # Generate basic documentation from description and parameters
        lines = [f"**{self.name}** - {self.description}"]
        if self.parameters:
            for param_name, param_def in self.parameters.items():
                param_type = param_def.get("type", "any")
                param_desc = param_def.get("description", "")
                lines.append(f"  - `{param_name}` ({param_type}): {param_desc}")
        return "\n".join(lines)


@dataclass
class EncodedImage:
    """Base64-encoded image with MIME type.

    Attributes:
        data: Base64-encoded image data
        mime_type: MIME type (e.g., 'image/png', 'image/jpeg')
    """

    data: str
    mime_type: str

    def to_data_url(self) -> str:
        """Convert to data URL format for API calls."""
        return f"data:{self.mime_type};base64,{self.data}"


@beartype
def encode_image(image: Image.Image, quality: int = 85) -> EncodedImage:
    """Encode PIL Image to base64 with correct MIME type.

    Args:
        image: PIL Image to convert
        quality: JPEG quality (1-100), only used for RGB images

    Returns:
        EncodedImage with base64 data and MIME type
    """
    with io.BytesIO() as buffer:
        # Preserve grayscale mode for medical images
        if image.mode in ("L", "I", "F"):
            image.save(buffer, format="PNG", optimize=True)
            mime_type = "image/png"
        elif image.mode == "RGB":
            image.save(buffer, format="JPEG", quality=quality)
            mime_type = "image/jpeg"
        else:
            # RGBA, P, etc. - convert to RGB for JPEG
            image.convert("RGB").save(buffer, format="JPEG", quality=quality)
            mime_type = "image/jpeg"
        data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return EncodedImage(data=data, mime_type=mime_type)


class ToolRegistry:
    """Registry of tools available for agentic analysis.

    Supports async context manager protocol for proper resource cleanup:
        async with ToolRegistry(image_path) as registry:
            result = await registry.execute("zoom", factor=2.0)

    Tools can be registered manually or via factory functions like
    create_visual_tools() and create_search_tools().
    """

    @beartype
    def __init__(
        self,
        image_path: Path | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        """Initialize tool registry.

        Args:
            image_path: Path to the source image for visual tool operations
            tools: List of tools to register (use factory functions)
        """
        self._tools: dict[str, Tool] = {}
        self._image_path = image_path
        self._current_image: Image.Image | None = None
        self._tool_history: list[ToolResult] = []
        self._image_lock = asyncio.Lock()

        # Register provided tools
        for tool in tools or []:
            self.register(tool)

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

    def transform_image(self, operation: Callable[[Image.Image], Image.Image]) -> None:
        """Apply a transformation to the current image with automatic cleanup.

        Args:
            operation: Function that takes current image and returns new image

        Raises:
            ToolExecutionError: If no image is loaded

        Example:
            registry.transform_image(lambda img: zoom_image(img, 2.0))
        """
        if self._current_image is None:
            raise ToolExecutionError("No image loaded")
        old_image = self._current_image
        self._current_image = operation(old_image)
        old_image.close()

    @beartype
    def register(self, tool: Tool) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool to register
        """
        self._tools[tool.name] = tool

    @beartype
    def set_image(self, image_path: Path) -> None:
        """Set the source image for tool operations.

        Args:
            image_path: Path to the image file
        """
        if self._current_image is not None:
            self._current_image.close()
        self._image_path = image_path
        with Image.open(image_path) as img:
            self._current_image = img.copy()
        self._tool_history.clear()

    @beartype
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for all registered tools.

        Returns:
            List of tool schemas in OpenAI function-calling format
        """
        schemas = []
        for tool in self._tools.values():
            properties = {}
            required_params = []

            for param_name, param_def in tool.parameters.items():
                # Preserve common JSON Schema validation keywords to keep the model
                # inside the executable parameter bounds. Type must be explicit to
                # avoid silent schema drift.
                param_type = param_def.get("type")
                if not param_type:
                    raise ValueError(
                        f"Tool '{tool.name}' parameter '{param_name}' is missing required 'type'"
                    )
                prop: dict[str, Any] = {"type": param_type}
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
                        prop[key] = param_def[key]

                if param_def.get("type") == "array" and "items" in param_def:
                    prop["items"] = param_def["items"]

                # Required when no default is provided.
                if "default" not in param_def:
                    required_params.append(param_name)

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
        """Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with execution outcome

        Raises:
            UnknownToolError: If tool_name is not registered
            ToolExecutionError: If tool requires image but none is loaded
        """
        if tool_name not in self._tools:
            raise UnknownToolError(tool_name, list(self._tools.keys()))

        tool = self._tools[tool_name]

        if tool.requires_image:
            await self._ensure_image_loaded()
            if self._current_image is None:
                raise ToolExecutionError(
                    f"Tool '{tool_name}' requires an image, but no image path was provided"
                )

        try:
            result = await tool.execute(self, **kwargs)
        except ToolExecutionError:
            raise
        except ValueError as e:
            raise ToolExecutionError(
                f"Tool '{tool_name}' received invalid arguments: {e}"
            ) from e
        except TypeError as e:
            raise ToolExecutionError(
                f"Tool '{tool_name}' call signature is invalid: {e}"
            ) from e
        # Let unexpected exceptions propagate - don't mask bugs with generic wrapping

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

    @current_image.setter
    def current_image(self, image: Image.Image | None) -> None:
        """Set the current image, closing the old one to prevent memory leaks.

        Note: Use transform_image() for most operations to ensure proper cleanup.
        This setter is primarily for reset operations where you have a new image.
        """
        if self._current_image is not None and self._current_image is not image:
            self._current_image.close()
        self._current_image = image

    @property
    def image_path(self) -> Path | None:
        """Get the path to the source image."""
        return self._image_path

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
    ) -> str:
        """Generate prompt documentation for all registered tools.

        This creates formatted text suitable for inclusion in system prompts,
        documenting all available tools with their parameters and usage.

        Args:
            group_by_category: If True, group tools by category with headers
            include_categories: If set, only include tools from these categories
            exclude_categories: If set, exclude tools from these categories

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
                categories = {
                    k: v for k, v in categories.items() if k not in exclude_categories
                }

            # Sort categories for consistent output
            for category in sorted(categories.keys()):
                tools = categories[category]
                if not tools:
                    continue

                # Category header
                category_title = category.replace("_", " ").title()
                sections.append(f"**{category_title} Tools:**\n")

                # Tool documentation
                for tool in sorted(tools, key=lambda t: t.name):
                    sections.append(tool.get_prompt_documentation())
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
                sections.append(tool.get_prompt_documentation())
                sections.append("")

        return "\n".join(sections).strip()

    @beartype
    def get_tool_names(self) -> list[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())

    @beartype
    def get_categories(self) -> set[str]:
        """Get set of all tool categories."""
        return {tool.category or "other" for tool in self._tools.values()}
