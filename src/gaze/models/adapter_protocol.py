"""Adapter protocol and types for model adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import runtime_checkable


@dataclass(frozen=True)
class GenerationLog:
    """Token usage metadata for a generation call."""

    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    reasoning_content: str | None = None

    @property
    def tokens(self) -> int:
        """Total tokens used."""
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class AdapterProtocol(Protocol):
    """Minimal interface required by AgenticProcessorBase.

    Declared ``@runtime_checkable`` so an adapter instance can be validated
    with ``isinstance`` (used by ``@beartype`` when an adapter is passed
    directly to ``AgenticProcessorBase``). ``isinstance`` checks only that the
    required members are present, not their signatures.
    """

    supports_multipart_tool_content: bool

    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        seed: int | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog] | AsyncIterator[str]: ...

    async def aclose(self) -> None:
        """Release resources held by the adapter (clients, models, caches)."""
        ...
