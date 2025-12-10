"""Tool bridge for verifiers integration.

Provides utilities for executing radiant_harness tools within
verifiers environments, handling format conversion and state management.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from beartype import beartype

from radiant_harness.tools import Tool
from radiant_harness.tools import ToolRegistry
from radiant_harness.tools import create_search_tools
from radiant_harness.tools import create_visual_tools
from radiant_harness.types import ToolResult


class ToolBridge:
    """Bridge for executing tools within verifiers environments.

    Handles:
    - Tool registry creation and lifecycle
    - Parsing tool requests from model outputs
    - Formatting tool results for verifiers messages
    - Image state management across turns
    """

    @beartype
    def __init__(
        self,
        enable_visual_tools: bool = True,
        enable_search_tools: bool = False,
        disabled_tools: set[str] | None = None,
        max_tool_calls_per_turn: int = 5,
    ) -> None:
        """Initialize tool bridge.

        Args:
            enable_visual_tools: Enable visual manipulation tools
            enable_search_tools: Enable web/image search tools
            disabled_tools: Specific tools to disable
            max_tool_calls_per_turn: Maximum tool calls per turn
        """
        self.enable_visual_tools = enable_visual_tools
        self.enable_search_tools = enable_search_tools
        self._disabled_tools = disabled_tools or set()
        self.max_tool_calls_per_turn = max_tool_calls_per_turn

        # Build tool list
        self._tools: list[Tool] = []
        if enable_visual_tools:
            self._tools.extend(create_visual_tools(self._disabled_tools))
        if enable_search_tools:
            self._tools.extend(create_search_tools(self._disabled_tools))

        # Active registry (created per-episode)
        self._registry: ToolRegistry | None = None

    @property
    def tool_names(self) -> list[str]:
        """Get list of available tool names."""
        return [t.name for t in self._tools]

    @beartype
    def create_registry(self, image_path: Path | None = None) -> ToolRegistry:
        """Create a new tool registry for an episode.

        Args:
            image_path: Path to the image for visual tools

        Returns:
            Configured ToolRegistry
        """
        self._registry = ToolRegistry(image_path=image_path, tools=self._tools)
        return self._registry

    @beartype
    async def close_registry(self) -> None:
        """Close and cleanup the active registry."""
        if self._registry is not None:
            await self._registry.aclose()
            self._registry = None

    @beartype
    def get_tool_documentation(self) -> str:
        """Get formatted tool documentation for prompts."""
        if not self._tools:
            return ""

        docs = []
        for tool in self._tools:
            param_info = []
            for pname, pschema in tool.parameters.get("properties", {}).items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                required = pname in tool.parameters.get("required", [])
                req_str = " (required)" if required else " (optional)"
                param_info.append(f"    - {pname}: {ptype}{req_str} - {pdesc}")

            params_str = "\n".join(param_info) if param_info else "    (no parameters)"
            docs.append(f"**{tool.name}**: {tool.description}\n{params_str}")

        return "\n\n".join(docs)

    @beartype
    def parse_tool_requests(
        self,
        text: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Parse tool requests from model output text.

        Supports multiple formats:
        - JSON: {"tool": "name", "args": {...}}
        - Inline: TOOL_NAME[arg1, arg2]
        - XML-style: <tool name="...">args</tool>

        Args:
            text: Model output text

        Returns:
            List of (tool_name, arguments) tuples
        """
        requests: list[tuple[str, dict[str, Any]]] = []

        # Try JSON format first
        json_requests = self._parse_json_tools(text)
        if json_requests:
            return json_requests[: self.max_tool_calls_per_turn]

        # Try inline format
        inline_requests = self._parse_inline_tools(text)
        if inline_requests:
            return inline_requests[: self.max_tool_calls_per_turn]

        return requests

    @beartype
    def _parse_json_tools(self, text: str) -> list[tuple[str, dict[str, Any]]]:
        """Parse JSON-formatted tool requests."""
        requests: list[tuple[str, dict[str, Any]]] = []

        # Find all JSON objects in text
        json_pattern = r'\{[^{}]*"tool"[^{}]*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                obj = json.loads(match.group())
                tool_name = obj.get("tool") or obj.get("name")
                args = obj.get("args") or obj.get("arguments") or {}

                if tool_name and tool_name in self.tool_names:
                    requests.append((tool_name, args))
            except json.JSONDecodeError:
                continue

        return requests

    @beartype
    def _parse_inline_tools(self, text: str) -> list[tuple[str, dict[str, Any]]]:
        """Parse inline TOOL[args] formatted requests."""
        requests: list[tuple[str, dict[str, Any]]] = []

        for tool_name in self.tool_names:
            # Match inline tool calls like: tool_name[args] or tool_name(args)
            pattern = rf"{tool_name}\s*[\[\(]([^\]\)]*?)[\]\)]"
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                args_str = match.group(1).strip()
                args = self._parse_inline_args(tool_name, args_str)
                requests.append((tool_name, args))

        return requests

    @beartype
    def _parse_inline_args(
        self,
        tool_name: str,
        args_str: str,
    ) -> dict[str, Any]:
        """Parse inline arguments into a dict.

        Maps positional arguments to parameter names based on tool schema.
        """
        if not args_str:
            return {}

        # Find the tool to get parameter order
        tool = next((t for t in self._tools if t.name == tool_name), None)
        if tool is None:
            return {}

        # Get parameter names in order
        properties = tool.parameters.get("properties", {})
        param_names = list(properties.keys())

        # Split arguments
        parts = [p.strip() for p in args_str.split(",")]

        # Map to parameter names
        args: dict[str, Any] = {}
        for i, part in enumerate(parts):
            if i >= len(param_names):
                break

            param_name = param_names[i]
            param_schema = properties.get(param_name, {})
            param_type = param_schema.get("type", "string")

            # Convert type
            value: Any = part
            if param_type == "number":
                try:
                    value = float(part)
                except ValueError:
                    value = part
            elif param_type == "integer":
                try:
                    value = int(float(part))
                except ValueError:
                    value = part
            elif param_type == "boolean":
                value = part.lower() in ("true", "1", "yes")

            args[param_name] = value

        return args

    @beartype
    async def execute_tools(
        self,
        requests: list[tuple[str, dict[str, Any]]],
    ) -> list[ToolResult]:
        """Execute a list of tool requests.

        Args:
            requests: List of (tool_name, arguments) tuples

        Returns:
            List of ToolResult objects

        Raises:
            RuntimeError: If registry not created
        """
        if self._registry is None:
            raise RuntimeError("Tool registry not created. Call create_registry first.")

        results: list[ToolResult] = []
        for tool_name, args in requests:
            result = await self._registry.execute(tool_name, **args)
            results.append(result)

        return results

    @beartype
    def format_results_for_verifiers(
        self,
        results: list[ToolResult],
    ) -> list[dict[str, Any]]:
        """Format tool results for verifiers messages.

        Args:
            results: List of tool execution results

        Returns:
            List of message dicts for verifiers
        """
        messages: list[dict[str, Any]] = []

        for i, result in enumerate(results):
            content: str | list[dict[str, Any]]

            # Check if result has image
            image_url = result.get_image_data_url()
            if image_url:
                content = [
                    {"type": "text", "text": result.description},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            else:
                text_parts = [result.description]
                if result.error:
                    text_parts.append(f"Error: {result.error}")
                if formatted := result.formatted_results:
                    text_parts.append(formatted)
                content = "\n".join(text_parts)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"tool_{i}",
                    "content": content,
                }
            )

        return messages


@beartype
def create_tool_bridge(
    visual_tools: bool = True,
    search_tools: bool = False,
    disabled: set[str] | None = None,
) -> ToolBridge:
    """Create a configured tool bridge.

    Convenience function for creating a ToolBridge with common settings.

    Args:
        visual_tools: Enable visual manipulation tools
        search_tools: Enable web/image search tools
        disabled: Specific tools to disable

    Returns:
        Configured ToolBridge instance
    """
    return ToolBridge(
        enable_visual_tools=visual_tools,
        enable_search_tools=search_tools,
        disabled_tools=disabled,
    )
