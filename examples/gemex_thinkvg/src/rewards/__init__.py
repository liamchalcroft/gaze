"""Verifiable reward functions for GEMeX-ThinkVG RL training.

Provides three verifiable reward components:
1. Answer reward - semantic matching of medical findings
2. Location reward - anatomical region matching
3. BBox reward - IoU-based bounding box accuracy

Also provides GEMeXVerifiersReward for gaze verifiers integration.
"""

from __future__ import annotations

from .answer import compute_answer_reward
from .bbox import compute_bbox_reward
from .combined import GEMeXRewardFunction
from .combined import GEMeXVerifiersReward
from .combined import RewardWeights
from .combined import compute_combined_reward
from .location import compute_location_reward

__all__ = [
    "compute_answer_reward",
    "compute_location_reward",
    "compute_bbox_reward",
    "compute_combined_reward",
    "GEMeXRewardFunction",
    "GEMeXVerifiersReward",
    "RewardWeights",
]
