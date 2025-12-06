"""Exceptions for the radiology VLM agent harness."""

from __future__ import annotations

from typing import Any


class HarnessError(Exception):
    """Base exception for all harness errors."""


class ToolExecutionError(HarnessError):
    """Raised when a tool execution fails due to invalid state or parameters."""


class UnknownToolError(HarnessError):
    """Raised when an unknown tool is requested.

    Attributes:
        tool_name: The name of the unknown tool
        available_tools: List of available tool names
    """

    def __init__(self, tool_name: str, available_tools: list[str]) -> None:
        self.tool_name = tool_name
        self.available_tools = available_tools
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


class SchemaValidationError(HarnessError):
    """Raised when a response fails schema validation.

    Attributes:
        missing_fields: List of missing required fields
        response: The invalid response
    """

    def __init__(
        self,
        message: str,
        missing_fields: list[str] | None = None,
        response: dict[str, Any] | None = None,
    ) -> None:
        self.missing_fields = missing_fields or []
        self.response = response
        super().__init__(message)


class ModelError(HarnessError):
    """Base exception for model-related errors."""

    def __init__(self, message: str, model_name: str | None = None) -> None:
        super().__init__(message)
        self.model_name = model_name


class APIError(ModelError):
    """Raised when API calls fail."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message, model_name)
        self.status_code = status_code
        self.response_body = response_body
