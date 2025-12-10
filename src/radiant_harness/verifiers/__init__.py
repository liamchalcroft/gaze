"""Verifiers integration utilities for Radiant Harness.

Provides base classes and utilities for seamless integration with the
verifiers package for multi-turn RL training with verifiable rewards.

Key components:
- VerifiableProcessorMixin: Add verifiers support to processors
- RadiantHarnessAdapter: Bridge between harness and verifiers formats
- BaseMultiTurnEnv: Standard multi-turn environment template
- Reward functions: ExactMatch, TokenF1, IoU, Combined
"""

from __future__ import annotations

from .adapter import RadiantHarnessAdapter
from .adapter import create_verifiers_rubric
from .adapter import wrap_processor_for_verifiers
from .base import BaseMultiTurnEnv
from .mixin import VerifiableProcessorMixin
from .mixin import create_verifiable_processor
from .rewards import BaseRewardFunction
from .rewards import CombinedReward
from .rewards import ExactMatchReward
from .rewards import IoUReward
from .rewards import TokenF1Reward
from .tool_bridge import ToolBridge
from .tool_bridge import create_tool_bridge

__all__ = [
    # Mixin and utilities
    "VerifiableProcessorMixin",
    "create_verifiable_processor",
    # Adapter
    "RadiantHarnessAdapter",
    "create_verifiers_rubric",
    "wrap_processor_for_verifiers",
    # Tool bridge
    "ToolBridge",
    "create_tool_bridge",
    # Base environment
    "BaseMultiTurnEnv",
    # Reward functions
    "BaseRewardFunction",
    "ExactMatchReward",
    "TokenF1Reward",
    "IoUReward",
    "CombinedReward",
]
