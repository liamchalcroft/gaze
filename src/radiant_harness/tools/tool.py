"""Tool class definition for agentic processing."""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any

from beartype import beartype

if TYPE_CHECKING:
    from radiant_harness.types import ToolResult

# Type alias for tool execute functions
ToolExecutor = Callable[..., Awaitable["ToolResult"]]


@beartype
class Tool:
    """Tool definition for agentic processing."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        execute: ToolExecutor,
        requires_image: bool = False,
        category: str | None = None,
        prompt_documentation: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.execute = execute
        self.requires_image = requires_image
        self.category = category
        self._prompt_documentation = prompt_documentation

    def get_prompt_documentation(self) -> str:
        """Generate documentation for prompt inclusion.

        Returns custom prompt_documentation if provided, otherwise generates
        documentation from the tool's description and parameters.
        """
        if self._prompt_documentation:
            return self._prompt_documentation

        doc = f"**{self.name}**: {self.description}\n"
        if self.parameters:
            doc += "Parameters:\n"
            for param, info in self.parameters.items():
                required = "Required" if "default" not in info else "Optional"
                param_type = info.get("type", "unknown")
                param_desc = info.get("description", "")
                doc += f"- {param} ({param_type}, {required}): {param_desc}\n"
        return doc


__all__ = ["Tool", "ToolExecutor"]
