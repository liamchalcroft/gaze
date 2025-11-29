"""Tool filtering utilities for ablation studies.

Allows selectively enabling/disabling tools for experimental evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nova_retrieval_vlm.agentic.tools import ToolRegistry


@dataclass
class ToolFilterConfig:
    """Configuration for filtering tools."""

    enabled_tools: list[str] | None = None  # If specified, only these tools are available
    disabled_tools: list[str] | None = None  # If specified, these tools are disabled


class FilteredToolRegistry(ToolRegistry):
    """Tool registry with filtering support for ablation studies."""

    def __init__(self, image_path=None, filter_config: ToolFilterConfig | None = None):
        """Initialize with optional tool filtering."""
        self.filter_config = filter_config or ToolFilterConfig()
        super().__init__(image_path)

    def _register_default_tools(self) -> None:
        """Register default tools with filtering applied."""
        # Call parent registration
        super()._register_default_tools()

        # Apply filtering if configured
        self._apply_tool_filtering()

    def _apply_tool_filtering(self) -> None:
        """Apply tool filtering to the registered tools."""
        if self.filter_config.enabled_tools is not None:
            # Keep only enabled tools
            enabled_set = set(self.filter_config.enabled_tools)
            tools_to_remove = [name for name in self._tools if name not in enabled_set]
            for tool_name in tools_to_remove:
                del self._tools[tool_name]

        elif self.filter_config.disabled_tools is not None:
            # Remove disabled tools
            disabled_set = set(self.filter_config.disabled_tools)
            for tool_name in disabled_set:
                if tool_name in self._tools:
                    del self._tools[tool_name]

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get filtered tool schemas."""
        schemas = super().get_tool_schemas()

        # Add filtering information for research
        if self.filter_config.enabled_tools:
            for schema in schemas:
                schema["filter_info"] = {"enabled_only": True}
        elif self.filter_config.disabled_tools:
            for schema in schemas:
                schema["filter_info"] = {"disabled": self.filter_config.disabled_tools}

        return schemas

    def get_filter_summary(self) -> dict[str, Any]:
        """Get summary of current tool filtering."""
        return {
            "enabled_tools": list(self._tools.keys()),
            "disabled_tools": self.filter_config.disabled_tools or [],
            "whitelist_mode": self.filter_config.enabled_tools is not None,
            "total_tools_available": len(self._tools),
        }


def create_filtered_registry(
    image_path, enabled_tools: list[str] | None = None, disabled_tools: list[str] | None = None
) -> FilteredToolRegistry:
    """Create a filtered tool registry."""
    filter_config = ToolFilterConfig(enabled_tools=enabled_tools, disabled_tools=disabled_tools)
    return FilteredToolRegistry(image_path, filter_config)
