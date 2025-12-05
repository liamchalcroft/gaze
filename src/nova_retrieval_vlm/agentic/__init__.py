"""Agentic processing module for NOVA retrieval VLM.

Provides tool-calling capabilities, visual reasoning integration,
and multi-turn refinement for medical image analysis.
"""

from __future__ import annotations

from nova_retrieval_vlm.agentic.diagnosis import AgenticDiagnosisProcessor
from nova_retrieval_vlm.agentic.localization import AgenticLocalizationProcessor
from nova_retrieval_vlm.agentic.processor import AgenticProcessor
from nova_retrieval_vlm.agentic.processor import AgenticResult
from nova_retrieval_vlm.agentic.processor import ToolCall
from nova_retrieval_vlm.agentic.processor import Turn
from nova_retrieval_vlm.agentic.tools import EncodedImage
from nova_retrieval_vlm.agentic.tools import ToolExecutionError
from nova_retrieval_vlm.agentic.tools import ToolRegistry
from nova_retrieval_vlm.agentic.tools import ToolResult
from nova_retrieval_vlm.agentic.tools import UnknownToolError
from nova_retrieval_vlm.agentic.tools import VisualTool
from nova_retrieval_vlm.agentic.tools import encode_image

__all__ = [
    "AgenticDiagnosisProcessor",
    "AgenticLocalizationProcessor",
    "AgenticProcessor",
    "AgenticResult",
    "EncodedImage",
    "ToolCall",
    "Turn",
    "ToolExecutionError",
    "ToolRegistry",
    "ToolResult",
    "UnknownToolError",
    "VisualTool",
    "encode_image",
]
