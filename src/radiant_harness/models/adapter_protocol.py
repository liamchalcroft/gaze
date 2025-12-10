"""Lightweight protocol for chat adapters used by the harness."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol

if TYPE_CHECKING:
    from radiant_harness.models._types import GenerationLog


class AdapterProtocol(Protocol):
    """Minimal interface required by AgenticProcessorBase."""

    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        stream: bool = False,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog] | AsyncIterator[str]: ...
