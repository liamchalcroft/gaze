"""Verifiers integration utilities for GAZE.

First-class integration with the verifiers package for multi-turn RL training.

Key components:
- VerifiableProcessorMixin: Add verifiers support to any processor
- BaseMultiTurnEnv: Standard multi-turn environment template
- Reward functions: ExactMatch, TokenF1, IoU, Combined

``GazeAdapter``, ``BaseMultiTurnEnv``, and ``VerifiableProcessorMixin``
require the ``verifiers`` and ``datasets`` packages at runtime. They are lazily
imported so that ``from gaze.verifiers import ExactMatchReward`` (and
the other reward utilities) works without those heavy optional dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .rewards import BaseRewardFunction
from .rewards import CombinedReward
from .rewards import ExactMatchReward
from .rewards import IoUReward
from .rewards import TokenF1Reward
from .rewards import extract_completion_text

if TYPE_CHECKING:
    from .adapter import GazeAdapter
    from .base import BaseMultiTurnEnv
    from .mixin import VerifiableProcessorMixin

__all__ = [
    # Core mixin for adding verifiers support
    "VerifiableProcessorMixin",
    # Environment base class
    "BaseMultiTurnEnv",
    # Adapter utilities
    "GazeAdapter",
    # Reward functions
    "BaseRewardFunction",
    "ExactMatchReward",
    "TokenF1Reward",
    "IoUReward",
    "CombinedReward",
    # Utilities
    "extract_completion_text",
]


def __getattr__(name: str):
    """Lazy import for verifiers-dependent symbols.

    ``GazeAdapter``, ``BaseMultiTurnEnv``, and
    ``VerifiableProcessorMixin`` pull in ``verifiers`` and ``datasets``
    at import time.  We defer those imports so that lightweight consumers
    (e.g. reward functions only) never pay the cost.
    """
    if name == "GazeAdapter":
        from .adapter import GazeAdapter

        return GazeAdapter
    if name == "BaseMultiTurnEnv":
        from .base import BaseMultiTurnEnv

        return BaseMultiTurnEnv
    if name == "VerifiableProcessorMixin":
        from .mixin import VerifiableProcessorMixin

        return VerifiableProcessorMixin
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
