"""Tool registry with image management, execution tracking, and documentation.

This module provides:
- ToolDocumenter: Schema generation and documentation formatting
- ToolRegistry: Complete tool management with image handling and execution
- EncodedImage: Container for base64-encoded image data
- encode_image: Utility to encode PIL Images
"""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import Any

from beartype import beartype
from PIL import Image

from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.exceptions import UnknownToolError
from radiant_harness.tools.image_manager import ImageManager
from radiant_harness.tools.tool import Tool
from radiant_harness.types import ToolResult

# Valid JSON Schema types for tool parameters
VALID_PARAM_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}


@dataclass(frozen=True)
class EncodedImage:
    """Container for encoded image data."""

    data: str
    mime_type: str

    def to_data_url(self) -> str:
        """Convert to a data URL for embedding in HTML/messages."""
        return f"data:{self.mime_type};base64,{self.data}"


@beartype
def encode_image(image: Image.Image) -> EncodedImage:
    """Encode a PIL Image to base64 PNG string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    image_bytes = buffer.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return EncodedImage(data=image_base64, mime_type="image/png")


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
                if param_type not in VALID_PARAM_TYPES:
                    raise ValueError(
                        f"Tool '{tool.name}' parameter '{param_name}' has invalid type '{param_type}'. "
                        f"Must be one of: {', '.join(sorted(VALID_PARAM_TYPES))}"
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
                        prop[key] = param_def[key]

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
                categories = {k: v for k, v in categories.items() if k not in exclude_categories}

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
    ) -> None:
        """Initialize refactored tool registry."""
        # Initialize specialized managers
        self._image_manager = ImageManager()
        self._documenter = ToolDocumenter(tools)

        # Tool execution history
        self._tool_history: deque[ToolResult] = deque(maxlen=max_history)
        self.max_history = max_history

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
        """Close and clean up all resources."""
        self._image_manager.close()
        self._tool_history.clear()

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
        """Execute a tool by name with given arguments."""
        tool = self._documenter.get_tool(tool_name)
        if tool is None:
            raise UnknownToolError(tool_name, self._documenter.get_tool_names())

        if tool.requires_image:
            await self._image_manager.ensure_loaded()
            if not self._image_manager.has_image:
                raise ToolExecutionError(
                    f"Tool '{tool_name}' requires an image, but no image path was provided"
                )

        result = await tool.execute(self, **kwargs)

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
