"""Tool decorator system for simplified tool creation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from radiant_harness.tools.registry import Tool


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    requires_image: bool = False,
    category: str | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator to create a Tool from a function.

    Args:
        name: Tool name
        description: Tool description
        parameters: JSON schema for parameters
        requires_image: Whether tool requires an image
        category: Tool category for organization

    Returns:
        Decorator function that returns a Tool instance
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        """Create a Tool instance from the decorated function."""
        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            execute=func,
            requires_image=requires_image,
            category=category,
        )

    return decorator


def visual_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator for visual tools.

    Shortcut for @tool with requires_image=True and category="visual".
    """
    return tool(
        name=name,
        description=description,
        parameters=parameters,
        requires_image=True,
        category="visual",
    )
