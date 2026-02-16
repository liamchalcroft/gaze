"""Adapter protocol and types for model adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from typing import Protocol


@dataclass(frozen=True)
class GenerationLog:
    """Token usage metadata for a generation call."""

    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None

    @property
    def tokens(self) -> int:
        """Total tokens used."""
        return self.prompt_tokens + self.completion_tokens


class AdapterProtocol(Protocol):
    """Minimal interface required by AgenticProcessorBase."""

    supports_multipart_tool_content: bool

    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        stream: bool = False,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog] | AsyncIterator[str]: ...
