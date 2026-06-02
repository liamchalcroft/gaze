"""GAZE: a grounded agentic framework for vision-language models.

Subclass ``AgenticProcessorBase``, implement four methods, and call
``analyze()`` to run multi-turn agentic analysis with visual tools and
web search.

Example::

    from gaze import AgenticProcessorBase

    class MyProcessor(AgenticProcessorBase):
        def get_system_prompt(self, images, metadata):
            return "You are a medical imaging expert."

        def get_user_message(self, images, metadata):
            return f"Analyze this scan. History: {metadata.get('history', '')}"

        def get_response_schema(self):
            return None

        def validate_response(self, response):
            return "findings" in response

    processor = MyProcessor(model_name="openai/gpt-4o", use_tools=True)
    result = await processor.analyze(images=Path("scan.jpg"), metadata={})
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gaze.base import AgenticProcessorBase
from gaze.base import ImageInput
from gaze.config import CacheConfig
from gaze.config import GazeConfig
from gaze.config import ImageProcessingConfig
from gaze.config import SearchConfig
from gaze.config import config_context
from gaze.config import get_config
from gaze.config import reset_config
from gaze.config import set_config
from gaze.exceptions import AgenticProcessingError
from gaze.exceptions import APIError
from gaze.exceptions import GazeError
from gaze.exceptions import ModelError
from gaze.exceptions import SchemaValidationError
from gaze.exceptions import TemplateError
from gaze.exceptions import ToolExecutionError
from gaze.exceptions import UnknownToolError
from gaze.models import AdapterProtocol
from gaze.models import GenerationLog
from gaze.models import LMStudioAdapter
from gaze.models import OpenAIAdapter
from gaze.models import list_lmstudio_model_ids
from gaze.models import require_lmstudio_model

if TYPE_CHECKING:
    from gaze.models.huggingface_adapter import HuggingFaceAdapter
    from gaze.models.huggingface_adapter import HuggingFaceVLMAdapter

# HuggingFace adapters are lazily imported to avoid torch dependency
# Use: from gaze import HuggingFaceAdapter, HuggingFaceVLMAdapter
from gaze.prompts import AnalysisMode
from gaze.prompts import combine_prompts
from gaze.prompts import create_prompt
from gaze.prompts import load_prompt
from gaze.prompts import load_template
from gaze.tools import EncodedImage
from gaze.tools import Tool
from gaze.tools import ToolRegistry
from gaze.tools import create_search_tools
from gaze.tools import create_visual_tools
from gaze.tools import encode_image
from gaze.types import AgenticResult
from gaze.types import RunConfig
from gaze.types import ToolCall
from gaze.types import ToolResult
from gaze.types import Turn
from gaze.utils.json_coerce import coerce_json_types

__version__ = "0.1.0"

__all__ = [
    # Core
    "AgenticProcessorBase",
    "ImageInput",
    # Configuration
    "GazeConfig",
    "CacheConfig",
    "SearchConfig",
    "ImageProcessingConfig",
    "config_context",
    "get_config",
    "reset_config",
    "set_config",
    # Result types
    "AgenticResult",
    "RunConfig",
    "Turn",
    "ToolCall",
    "ToolResult",
    # Tools
    "Tool",
    "ToolRegistry",
    "EncodedImage",
    "create_visual_tools",
    "create_search_tools",
    "encode_image",
    # Adapters
    "OpenAIAdapter",
    "LMStudioAdapter",
    "list_lmstudio_model_ids",
    "require_lmstudio_model",
    "HuggingFaceAdapter",
    "HuggingFaceVLMAdapter",
    "GenerationLog",
    "AdapterProtocol",
    # Prompts
    "AnalysisMode",
    "load_template",
    "load_prompt",
    "create_prompt",
    "combine_prompts",
    # Utilities
    "coerce_json_types",
    # Exceptions
    "GazeError",
    "AgenticProcessingError",
    "ToolExecutionError",
    "UnknownToolError",
    "SchemaValidationError",
    "TemplateError",
    "ModelError",
    "APIError",
]


def __getattr__(name: str):
    """Lazy import for HuggingFace adapters to avoid torch dependency."""
    if name == "HuggingFaceAdapter":
        from gaze.models.huggingface_adapter import HuggingFaceAdapter

        return HuggingFaceAdapter
    if name == "HuggingFaceVLMAdapter":
        from gaze.models.huggingface_adapter import HuggingFaceVLMAdapter

        return HuggingFaceVLMAdapter
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
