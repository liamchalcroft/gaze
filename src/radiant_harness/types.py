"""Core types for the radiology VLM agent harness.

Provides dataclasses for tool calls, results, conversation turns,
and agentic analysis results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from types import MappingProxyType
from typing import Any
from typing import Literal

# Valid roles for conversation turns
TurnRole = Literal["user", "assistant", "tool_result"]


@dataclass(frozen=True)
class ToolCall:
    """Represents a tool call request from the model.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool to execute
        arguments: JSON object or JSON string containing tool arguments
    """

    id: str
    name: str
    arguments: dict[str, Any] | str


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a tool.

    Attributes:
        tool_name: Name of the tool that was executed
        description: Human-readable description of what happened
        error: Error message if execution failed, None if successful
        image_base64: Base64-encoded image data if tool produced an image
        image_mime_type: MIME type of the image (e.g., 'image/png')
        metadata: Additional tool-specific metadata
    """

    tool_name: str
    description: str
    error: str | None = None
    image_base64: str | None = None
    image_mime_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        if bool(self.image_base64) != bool(self.image_mime_type):
            raise ValueError("image_base64 and image_mime_type must both be set or both be None")
        # Freeze the metadata dict to enforce immutability contract
        if isinstance(self.metadata, MappingProxyType):
            return
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def success(self) -> bool:
        """Tool succeeded if no error occurred."""
        return self.error is None

    def get_image_data_url(self) -> str | None:
        """Get data URL for image if present."""
        if self.image_base64 and self.image_mime_type:
            return f"data:{self.image_mime_type};base64,{self.image_base64}"
        return None

    @property
    def formatted_results(self) -> str | None:
        """Optional formatted result string stored in metadata."""
        formatted = self.metadata.get("formatted_results")
        return str(formatted) if formatted is not None else None


@dataclass(frozen=True)
class Turn:
    """Represents a single turn in the agentic conversation.

    Attributes:
        role: Message role - must be 'user', 'assistant', or 'tool_result'
        content: Text content of the turn
        tool_calls: List of tool calls made in this turn (assistant only)
        tool_results: List of tool results (tool_result only)
        image_base64: Optional image data for this turn
    """

    role: TurnRole
    content: str
    tool_calls: list[ToolCall] = field(default_factory=lambda: [])
    tool_results: list[ToolResult] = field(default_factory=lambda: [])
    image_base64: str | None = None


@dataclass(frozen=True)
class AgenticResult:
    """Result of an agentic analysis session.

    Attributes:
        final_response: Complete JSON response from the model
        turns: All conversation turns in the analysis
        total_tokens: Total tokens consumed across all turns
        confidence: Overall confidence in the analysis (0.0-1.0)
    """

    final_response: dict[str, Any]
    turns: list[Turn]
    total_tokens: int
    confidence: float

    @property
    def num_turns(self) -> int:
        """Number of turns in the conversation."""
        return len(self.turns)

    @property
    def tool_call_count(self) -> int:
        """Total number of tool calls made."""
        return sum(len(t.tool_calls) for t in self.turns)

    def get_tools_used(self) -> set[str]:
        """Get set of unique tool names used in the analysis."""
        tools: set[str] = set()
        for turn in self.turns:
            for tc in turn.tool_calls:
                tools.add(tc.name)
        return tools
