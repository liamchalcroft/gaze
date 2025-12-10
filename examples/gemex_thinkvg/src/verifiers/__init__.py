"""Verifiers integration for GEMeX-ThinkVG RL training.

Provides multi-turn environment for agentic visual grounding with
verifiable rewards (answer, location, bbox).
"""

from __future__ import annotations

from .environment import GEMeXThinkVGToolEnv, load_environment

__all__ = [
    "GEMeXThinkVGToolEnv",
    "load_environment",
]
