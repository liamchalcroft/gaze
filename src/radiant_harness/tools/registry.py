"""Refactored tool registry with separated responsibilities.

This module provides a cleaner ToolRegistry that delegates to specialized
managers for different aspects of tool management.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING
from typing import Any

from beartype import beartype

from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.exceptions import UnknownToolError
from radiant_harness.tools.image_manager import ImageManager
from radiant_harness.tools.tool import Tool
from radiant_harness.tools.tool_documenter import ToolDocumenter
from radiant_harness.types import ToolResult


@dataclass(frozen=True)
class EncodedImage:
    """Container for encoded image data."""

    data: str
    mime_type: str

    def to_data_url(self) -> str:
        """Convert to a data URL for embedding in HTML/messages."""
        return f"data:{self.mime_type};base64,{self.data}"


@beartype
def encode_image(image) -> EncodedImage:
    """Encode a PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    image_bytes = buffer.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return EncodedImage(data=image_base64, mime_type="image/png")


if TYPE_CHECKING:
    pass


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
        self._tool_history: list[ToolResult] = []
        self.max_history = max_history

        # Set initial image if provided
        if image_path:
            self._image_manager.set_image(image_path)

    async def __aenter__(self) -> ToolRegistry:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit with cleanup."""
        await self.aclose()

    @beartype
    async def aclose(self) -> None:
        """Async close to properly clean up search engine sessions."""
        self._image_manager.close()
        self._tool_history.clear()

        # Close search managers if they were injected via tool execution
        for tool in self._documenter.get_all_tools():
            if hasattr(tool, "_search_manager") and tool._search_manager is not None:
                await tool._search_manager.close()

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
    def set_image(self, image_path: Path) -> None:
        """Set the source image for tool operations."""
        self._image_manager.set_image(image_path)
        self._tool_history.clear()

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

        try:
            result = await tool.execute(self, **kwargs)
        except ToolExecutionError:
            raise
        except ValueError as e:
            raise ToolExecutionError(f"Tool '{tool_name}' received invalid arguments: {e}") from e
        except TypeError as e:
            raise ToolExecutionError(f"Tool '{tool_name}' call signature is invalid: {e}") from e

        self._tool_history.append(result)
        # Maintain history size limit to prevent memory leaks
        if len(self._tool_history) > self.max_history:
            self._tool_history.pop(0)
        return result

    @property
    def history(self) -> list[ToolResult]:
        """Get the history of tool executions."""
        return self._tool_history.copy()

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
