"""Exceptions for the radiology VLM agent harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class HarnessError(Exception):
    """Base exception for all harness errors."""


class ToolExecutionError(HarnessError):
    """Raised when a tool execution fails due to invalid state or parameters.

    Attributes:
        tool_name: Name of the tool that failed (if known)
        tool_args: Arguments passed to the tool (if available)
    """

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args
        super().__init__(message)


class TemplateError(HarnessError):
    """Raised when template loading or rendering fails.

    Chain with ``raise TemplateError(...) from original`` so that
    ``__cause__`` preserves the root error automatically.

    Attributes:
        template_path: Path to the template that failed
    """

    def __init__(
        self,
        message: str,
        template_path: Path | str | None = None,
    ) -> None:
        self.template_path = template_path
        if template_path:
            message = f"{message} (template: {template_path})"
        super().__init__(message)


class UnknownToolError(HarnessError):
    """Raised when an unknown tool is requested.

    Attributes:
        tool_name: The name of the unknown tool
        available_tools: Frozen tuple of available tool names
    """

    def __init__(self, tool_name: str, available_tools: list[str]) -> None:
        self.tool_name = tool_name
        self.available_tools: tuple[str, ...] = tuple(available_tools)
        super().__init__(
            f"Unknown tool '{tool_name}'. Available tools: {', '.join(sorted(available_tools))}"
        )


class AgenticProcessingError(HarnessError):
    """Raised when agentic processing fails.

    Attributes:
        turns_completed: Number of turns completed before failure
        partial_response: Partial response if available
    """

    def __init__(
        self,
        message: str,
        turns_completed: int,
        partial_response: dict[str, Any] | None = None,
    ) -> None:
        self.turns_completed = turns_completed
        self.partial_response = partial_response
        super().__init__(message)


class SchemaValidationError(AgenticProcessingError):
    """Raised when a response fails schema validation.

    Attributes:
        turns_completed: Number of turns completed before the invalid response
        missing_fields: List of missing required fields
        response: The invalid response
    """

    def __init__(
        self,
        message: str,
        turns_completed: int,
        missing_fields: list[str] | None = None,
        response: dict[str, Any] | None = None,
    ) -> None:
        self.missing_fields = missing_fields or []
        self.response = response
        super().__init__(message, turns_completed=turns_completed, partial_response=response)


class ModelError(HarnessError):
    """Base exception for model-related errors."""

    def __init__(self, message: str, model_name: str | None = None) -> None:
        super().__init__(message)
        self.model_name = model_name


class APIError(ModelError):
    """Raised when API calls fail.

    Note: This exception deliberately does NOT store the raw API response
    body.  OpenAI/OpenRouter error responses may echo back request headers
    or URLs that contain API keys.
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, model_name)
        self.status_code = status_code
