"""Radiant Harness -- agentic tool harness for vision-language models.

Subclass ``AgenticProcessorBase``, implement four methods, and call
``analyze()`` to run multi-turn agentic analysis with visual tools and
web search.

Example::

    from radiant_harness import AgenticProcessorBase

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

from radiant_harness.base import AgenticProcessorBase
from radiant_harness.base import ImageInput
from radiant_harness.config import AgenticConfig
from radiant_harness.config import CacheConfig
from radiant_harness.config import HarnessConfig
from radiant_harness.config import ImageProcessingConfig
from radiant_harness.config import RankingWeights
from radiant_harness.config import SearchConfig
from radiant_harness.config import config_context
from radiant_harness.config import get_config
from radiant_harness.config import reset_config
from radiant_harness.config import set_config
from radiant_harness.exceptions import AgenticProcessingError
from radiant_harness.exceptions import APIError
from radiant_harness.exceptions import HarnessError
from radiant_harness.exceptions import ModelError
from radiant_harness.exceptions import SchemaValidationError
from radiant_harness.exceptions import TemplateError
from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.exceptions import UnknownToolError
from radiant_harness.models import AdapterProtocol
from radiant_harness.models import GenerationLog
from radiant_harness.models import LMStudioAdapter
from radiant_harness.models import OpenAIAdapter
from radiant_harness.models import list_lmstudio_model_ids
from radiant_harness.models import require_lmstudio_model

if TYPE_CHECKING:
    from radiant_harness.models.huggingface_adapter import HuggingFaceAdapter
    from radiant_harness.models.huggingface_adapter import HuggingFaceVLMAdapter

# HuggingFace adapters are lazily imported to avoid torch dependency
# Use: from radiant_harness import HuggingFaceAdapter, HuggingFaceVLMAdapter
from radiant_harness.prompts import AnalysisMode
from radiant_harness.prompts import combine_prompts
from radiant_harness.prompts import create_prompt
from radiant_harness.prompts import load_prompt
from radiant_harness.prompts import load_system_prompt
from radiant_harness.prompts import load_task_prompt
from radiant_harness.prompts import load_template
from radiant_harness.tools import EncodedImage
from radiant_harness.tools import Tool
from radiant_harness.tools import ToolRegistry
from radiant_harness.tools import create_search_tools
from radiant_harness.tools import create_visual_tools
from radiant_harness.tools import encode_image
from radiant_harness.types import AgenticResult
from radiant_harness.types import ToolCall
from radiant_harness.types import ToolResult
from radiant_harness.types import Turn
from radiant_harness.types import TurnRole

__version__ = "0.1.0"

__all__ = [
    # Core
    "AgenticProcessorBase",
    "ImageInput",
    # Configuration
    "HarnessConfig",
    "AgenticConfig",
    "CacheConfig",
    "SearchConfig",
    "ImageProcessingConfig",
    "RankingWeights",
    "config_context",
    "get_config",
    "reset_config",
    "set_config",
    # Result types
    "AgenticResult",
    "Turn",
    "TurnRole",
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
    "load_system_prompt",
    "load_task_prompt",
    "create_prompt",
    "combine_prompts",
    # Exceptions
    "HarnessError",
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
        from radiant_harness.models.huggingface_adapter import HuggingFaceAdapter

        return HuggingFaceAdapter
    if name == "HuggingFaceVLMAdapter":
        from radiant_harness.models.huggingface_adapter import HuggingFaceVLMAdapter

        return HuggingFaceVLMAdapter
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
