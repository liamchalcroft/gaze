"""Tools module for GAZE.

Provides visual tools, search tools, and the tool registry.
"""

from __future__ import annotations

from gaze.tools.registry import EncodedImage
from gaze.tools.registry import ToolDocumenter
from gaze.tools.registry import ToolRegistry
from gaze.tools.registry import encode_image
from gaze.tools.search import create_search_tools
from gaze.tools.tool import Tool
from gaze.tools.visual import create_visual_tools

__all__ = [
    "EncodedImage",
    "Tool",
    "ToolDocumenter",
    "ToolRegistry",
    "create_search_tools",
    "create_visual_tools",
    "encode_image",
]
