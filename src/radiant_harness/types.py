"""Core types for the radiology VLM agent harness.

Provides dataclasses for tool calls, results, conversation turns,
and agentic analysis results.

All data types use ``frozen=True`` and immutable containers (tuples,
MappingProxyType) so that instances are safe to share across async
tasks and cannot be accidentally mutated after construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal

from radiant_harness._frozen import deep_freeze

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
    arguments: Mapping[str, Any] | str

    def __post_init__(self) -> None:
        if not isinstance(self.arguments, str):
            object.__setattr__(self, "arguments", deep_freeze(self.arguments))


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
    _data_url: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if bool(self.image_base64) != bool(self.image_mime_type):
            raise ValueError("image_base64 and image_mime_type must both be set or both be None")
        # Deep-freeze metadata even when callers pass a pre-wrapped proxy.
        object.__setattr__(self, "metadata", deep_freeze(self.metadata))
        # Pre-compute data URL once to avoid re-creating the large string
        # on every call (consistent with EncodedImage._data_url caching).
        if self.image_base64 and self.image_mime_type:
            object.__setattr__(
                self,
                "_data_url",
                f"data:{self.image_mime_type};base64,{self.image_base64}",
            )

    @property
    def success(self) -> bool:
        """Tool succeeded if no error occurred."""
        return self.error is None

    def get_image_data_url(self) -> str | None:
        """Get data URL for image if present."""
        return self._data_url

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
        tool_calls: Tool calls made in this turn (assistant only)
        tool_results: Tool results (tool_result only)
        image_base64: Optional image data for this turn
    """

    role: TurnRole
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    image_base64: str | None = None

    def __post_init__(self) -> None:
        # Coerce sequences to tuples to enforce immutability.
        # Callers may pass lists for convenience; we freeze them here.
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if not isinstance(self.tool_results, tuple):
            object.__setattr__(self, "tool_results", tuple(self.tool_results))


@dataclass(frozen=True)
class AgenticResult:
    """Result of an agentic analysis session.

    Attributes:
        final_response: Complete JSON response from the model
        turns: All conversation turns in the analysis
        total_tokens: Total tokens consumed across all turns
        confidence: Overall confidence in the analysis (0.0-1.0)
    """

    final_response: Mapping[str, Any]
    turns: tuple[Turn, ...]
    total_tokens: int
    confidence: float

    def __post_init__(self) -> None:
        # Deep-freeze mutable containers passed by callers.
        object.__setattr__(self, "final_response", deep_freeze(self.final_response))
        if not isinstance(self.turns, tuple):
            object.__setattr__(self, "turns", tuple(self.turns))

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
