"""GEMeX-ThinkVG example: RL fine-tuning with verifiable rewards.

This example demonstrates reinforcement learning fine-tuning using the
GAZE with multi-rollout generation and verifiable rewards
for medical visual grounding tasks.

The GEMeX-ThinkVG task has three verifiable outputs:
1. Answer correctness (semantic matching)
2. Location reference (anatomical region matching)
3. Bounding box accuracy (IoU-based)

Uses the verifiers package for RL training with custom reward functions
that leverage the harness's tool-augmented reasoning capabilities.
"""

from __future__ import annotations

from .dataset import GEMeXDataset
from .processor import GEMeXProcessor
from .rewards import GEMeXRewardFunction
from .rewards import RewardWeights
from .rewards import compute_combined_reward
from .verifiers import GEMeXThinkVGToolEnv
from .verifiers import load_environment

__all__ = [
    "GEMeXDataset",
    "GEMeXProcessor",
    "GEMeXRewardFunction",
    "RewardWeights",
    "compute_combined_reward",
    "GEMeXThinkVGToolEnv",
    "load_environment",
]
