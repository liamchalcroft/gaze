"""Model adapters for vision-language model APIs."""

from .base import BaseAdapter
from .openai_adapter import OpenAIAdapter
from .openrouter_adapter import OpenRouterAdapter


def get_model_client(model_name: str) -> OpenAIAdapter:
    """Get a model client for the specified model using OpenRouter interface.

    Args:
        model_name: Model identifier (OpenRouter format: provider/model:tier)

    Returns:
        Configured OpenAIAdapter client for OpenRouter
    """
    return OpenAIAdapter(model_name=model_name, max_retries=3, timeout=60)


__all__ = [
    "BaseAdapter",
    "OpenAIAdapter",
    "OpenRouterAdapter",
    "get_model_client",
]
