"""Verifiers integration utilities for Radiant Harness.

First-class integration with the verifiers package for multi-turn RL training.

Key components:
- VerifiableProcessorMixin: Add verifiers support to any processor
- BaseMultiTurnEnv: Standard multi-turn environment template
- Reward functions: ExactMatch, TokenF1, IoU, Combined
"""

from __future__ import annotations

from .adapter import RadiantHarnessAdapter
from .base import BaseMultiTurnEnv
from .mixin import VerifiableProcessorMixin
from .rewards import BaseRewardFunction
from .rewards import CombinedReward
from .rewards import ExactMatchReward
from .rewards import IoUReward
from .rewards import TokenF1Reward
from .rewards import extract_completion_text

__all__ = [
    # Core mixin for adding verifiers support
    "VerifiableProcessorMixin",
    # Environment base class
    "BaseMultiTurnEnv",
    # Adapter utilities
    "RadiantHarnessAdapter",
    # Reward functions
    "BaseRewardFunction",
    "ExactMatchReward",
    "TokenF1Reward",
    "IoUReward",
    "CombinedReward",
    # Utilities
    "extract_completion_text",
]
