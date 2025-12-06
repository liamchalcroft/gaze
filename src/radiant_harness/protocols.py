"""Protocols for the radiology VLM agent harness.

Defines the interfaces that task-specific implementations must satisfy.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import runtime_checkable

if TYPE_CHECKING:
    from radiant_harness.types import AgenticResult


@runtime_checkable
class TaskSchema(Protocol):
    """Protocol for task-specific output schemas.

    Implementations define the JSON schema for structured outputs
    and methods to validate responses.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this schema."""
        ...

    @property
    def json_schema(self) -> dict[str, Any]:
        """OpenAI-compatible JSON schema for structured outputs.

        Returns a dict with 'type': 'json_schema' and 'json_schema' keys.
        """
        ...

    def validate(self, response: dict[str, Any]) -> bool:
        """Validate that a response conforms to this schema.

        Args:
            response: Parsed JSON response from model

        Returns:
            True if valid, False otherwise
        """
        ...

    def get_required_fields(self) -> list[str]:
        """Get list of required top-level fields."""
        ...


@runtime_checkable
class TaskEvaluator(Protocol):
    """Protocol for task-specific evaluation.

    Implementations define how to evaluate model responses
    against ground truth for a specific task.
    """

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Evaluate predictions against ground truth.

        Args:
            predictions: List of model predictions (parsed JSON)
            ground_truth: List of ground truth labels (parsed JSON)

        Returns:
            Dictionary of metric names to values
        """
        ...


@runtime_checkable
class PromptBuilder(Protocol):
    """Protocol for task-specific prompt construction.

    Implementations define how to build system and user prompts
    for a specific task and domain.
    """

    def build_system_prompt(
        self,
        metadata: dict[str, Any],
        enable_tools: bool = True,
        enable_web_search: bool = False,
    ) -> str:
        """Build the system prompt for the task.

        Args:
            metadata: Task and image metadata
            enable_tools: Whether visual tools are enabled
            enable_web_search: Whether web search is enabled

        Returns:
            System prompt string
        """
        ...

    def build_user_message(
        self,
        image_path: Path,
        metadata: dict[str, Any],
    ) -> str:
        """Build the user message for the task.

        Args:
            image_path: Path to the image being analyzed
            metadata: Task and image metadata

        Returns:
            User message string
        """
        ...


@runtime_checkable
class ResponseParser(Protocol):
    """Protocol for parsing model responses into task-specific structures."""

    def parse(self, result: AgenticResult) -> dict[str, Any]:
        """Parse an AgenticResult into task-specific output.

        Args:
            result: Raw agentic result from processor

        Returns:
            Task-specific parsed response

        Raises:
            ValueError: If required fields are missing
        """
        ...
