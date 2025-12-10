"""Verifiers integration utilities for Radiant Harness.

First-class integration with the verifiers package for multi-turn RL training.

Key components:
- VerifiableProcessorMixin: Add verifiers support to any processor
- BaseMultiTurnEnv: Standard multi-turn environment template
- Reward functions: ExactMatch, TokenF1, IoU, Combined
- ToolBridge: Execute radiant_harness tools in verifiers environments
"""

from __future__ import annotations

from .adapter import RadiantHarnessAdapter
from .adapter import create_verifiers_rubric
from .base import BaseMultiTurnEnv
from .mixin import VerifiableProcessorMixin
from .mixin import create_verifiable_processor
from .rewards import BaseRewardFunction
from .rewards import CombinedReward
from .rewards import ExactMatchReward
from .rewards import IoUReward
from .rewards import TokenF1Reward
from .rewards import extract_completion_text
from .tool_bridge import ToolBridge
from .tool_bridge import create_tool_bridge

__all__ = [
    # Core mixin for adding verifiers support
    "VerifiableProcessorMixin",
    "create_verifiable_processor",
    # Environment base class
    "BaseMultiTurnEnv",
    # Adapter utilities
    "RadiantHarnessAdapter",
    "create_verifiers_rubric",
    # Tool execution bridge
    "ToolBridge",
    "create_tool_bridge",
    # Reward functions
    "BaseRewardFunction",
    "ExactMatchReward",
    "TokenF1Reward",
    "IoUReward",
    "CombinedReward",
    # Utilities
    "extract_completion_text",
]
