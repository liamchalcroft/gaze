"""Internal types for model adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationLog:
    """Token usage metadata for a generation call."""

    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None

    @property
    def tokens(self) -> int:
        """Total tokens used."""
        return self.prompt_tokens + self.completion_tokens
