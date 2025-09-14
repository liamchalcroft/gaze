from .base import BaseAdapter
from .openai_adapter import OpenAIAdapter


def get_model_client(model_name: str = "mistralai/mistral-small-3.2-24b-instruct:free"):
    """
    Get a model client for the specified model using OpenRouter interface.
    
    This uses the same OpenRouter interface as the main CLI tasks for consistency.
    
    Args:
        model_name: Name of the model to use (default: openai/gpt-4o)
        
    Returns:
        An OpenAIAdapter client configured for OpenRouter (same as CLI tasks)
    """
    # Always use OpenRouter interface for consistency with main CLI tasks
    return OpenAIAdapter(
        model_name=model_name,
        max_retries=3,
        timeout=60
        # base_url defaults to OpenRouter in OpenAIAdapter
    )

__all__ = ["base", "qwen_router", "internvlm_router", "get_model_client", "OpenAIAdapter", "BaseAdapter"] 