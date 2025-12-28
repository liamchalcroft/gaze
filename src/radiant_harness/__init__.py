"""Radiology VLM Agent Harness.

A modular framework for building multi-turn agentic vision-language model
systems for medical image analysis. Provides the core infrastructure for
tool-augmented reasoning over radiological images.

Key Features:
    - Multi-turn agentic analysis with configurable turn limits
    - Visual manipulation tools (zoom, crop, contrast, threshold, flip, rotate)
    - Web search integration for evidence-based analysis (PubMed, Open-i)
    - Structured JSON output with schema validation
    - Prompt templating via Jinja2
    - Extensible architecture via abstract base class

Quick Start:
    1. Subclass AgenticProcessorBase
    2. Implement the abstract methods for your task
    3. Create your prompts in a prompts/ directory
    4. Use the built-in tools or add custom ones

Example:
    ```python
    from harness import AgenticProcessorBase, ToolRegistry, create_visual_tools

    class MyRadiologyProcessor(AgenticProcessorBase):
        '''Custom processor for chest X-ray analysis.'''

        def get_system_prompt(self, image_path, metadata, width, height):
            return load_template(
                PROMPTS_DIR / "system.jinja",
                {"width": width, "height": height, **metadata}
            )

        def get_user_message(self, image_path, metadata):
            return f"Analyze this chest X-ray. History: {metadata.get('history', 'None')}"

        def get_response_schema(self):
            return {
                "type": "object",
                "properties": {
                    "findings": {"type": "array", "items": {"type": "string"}},
                    "impression": {"type": "string"},
                    "continue": {"type": "boolean"},
                },
                "required": ["findings", "impression", "continue"],
            }

        def validate_response(self, response):
            return "findings" in response and "impression" in response

    # Usage
    processor = MyRadiologyProcessor(
        model_name="openai/gpt-4o",
        use_tools=True,
        use_web_search=True,
        max_turns=10,
    )
    result = await processor.analyze(image_path, {"history": "Cough for 2 weeks"})
    print(result.final_response)
    ```

Architecture:
    The harness follows a dependency injection pattern where task-specific
    details (prompts, schemas, validation) are provided by subclasses while
    the core agentic loop, tool execution, and conversation management are
    handled by the base class.

    ```
    AgenticProcessorBase (abstract)
        |
        +-- get_system_prompt()      # Task-specific system prompt
        +-- get_user_message()       # Task-specific user message
        +-- get_response_schema()    # JSON schema for structured output
        +-- validate_response()      # Response validation
        +-- calculate_confidence()   # Optional: custom confidence scoring
        |
        +-- analyze()                # Main entry point (provided)
            +-- _run_analysis()      # Core agentic loop (provided)
            +-- _execute_tools()     # Tool execution (provided)
    ```

Tool System:
    Tools are registered via ToolRegistry and can be visual (image manipulation)
    or search-based (web/image search). The harness provides factory functions
    to create standard tool sets:

    - create_visual_tools(): zoom, crop, adjust_contrast, threshold, flip, rotate, reset
    - create_search_tools(): search_web (PubMed), search_images (Open-i)

    Custom tools can be added by creating Tool instances with an async execute function.

Response Format:
    The model's response must be valid JSON with a 'continue' field:
    - continue: true  -> Model wants another turn (will receive tool results or continue)
    - continue: false -> Model is done, response is final

    The harness enforces max_turns and provides warnings on the penultimate turn.

Modules:
    - base: AgenticProcessorBase abstract class
    - types: ToolCall, ToolResult, Turn, AgenticResult dataclasses
    - tools: ToolRegistry, Tool, create_visual_tools, create_search_tools
    - prompts: Jinja2 template loading utilities
    - exceptions: Harness-specific exception hierarchy
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
from radiant_harness.config import get_config
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
from radiant_harness.models import OpenAIAdapter

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
    "get_config",
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
