from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path

from beartype import beartype
from pydantic import BaseModel
from pydantic import Field


class GenerationLog(BaseModel):
    """Metadata about generation costs and tokens with comprehensive validation."""

    tokens: int = Field(ge=0, description="Number of tokens used in generation")
    cost: float = Field(ge=0.0, description="Cost of the API call in USD")
    model_name: str = Field(description="Name of the model used")
    timestamp: float = Field(description="Unix timestamp of generation")
    latency_seconds: float | None = Field(None, ge=0.0, description="Response latency in seconds")

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for logging/storage."""
        return self.model_dump()

    def cost_per_token(self) -> float:
        """Calculate cost per token."""
        return self.cost / self.tokens if self.tokens > 0 else 0.0


class BaseAdapter(ABC):
    """Abstract base class for vision-language model adapters."""

    @beartype
    def __init__(self, api_key: str) -> None:
        """Initialize adapter with API key."""
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        self.api_key = api_key

    @abstractmethod
    @beartype
    async def generate(
        self,
        image_path: Path,
        passages: Sequence[str],
        system_prompt: str | None = None,
    ) -> tuple[str, GenerationLog]:
        """
        Generate a completion using the model.

        Args:
            image_path: Path to the image file.
            passages: Retrieved text passages for context.
            system_prompt: Optional system prompt override.

        Returns:
            A tuple of (generated_text, GenerationLog).
        """
        raise NotImplementedError
