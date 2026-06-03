"""Tool class definition for agentic processing."""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

from beartype import beartype

if TYPE_CHECKING:
    from gaze.types import ToolResult


@beartype
@dataclass(frozen=True)
class Tool:
    """Tool definition for agentic processing.

    Immutable: a tool is constructed once (e.g. in ``create_visual_tools``)
    and never mutated, consistent with the frozen data types in ``gaze.types``.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[..., Awaitable[ToolResult]]
    requires_image: bool = False
    category: str | None = None
    prompt_documentation: str | None = None

    def get_prompt_documentation(self, *, compact: bool = False) -> str:
        """Generate documentation for prompt inclusion.

        Args:
            compact: If True, emit a single-line summary per tool to reduce
                token overhead for small-context models.

        Returns custom prompt_documentation if provided (unless compact),
        otherwise generates documentation from the tool's description and
        parameters.
        """
        if compact:
            params = ", ".join(
                f"{p}:{info.get('type', '?')}"
                for p, info in self.parameters.items()
                if "default" not in info
            )
            return f"- {self.name}({params}): {self.description}"

        if self.prompt_documentation:
            return self.prompt_documentation

        doc = f"**{self.name}**: {self.description}\n"
        if self.parameters:
            doc += "Parameters:\n"
            for param, info in self.parameters.items():
                required = "Required" if "default" not in info else "Optional"
                param_type = info.get("type", "unknown")
                param_desc = info.get("description", "")
                doc += f"- {param} ({param_type}, {required}): {param_desc}\n"
        return doc


__all__ = ["Tool"]
