"""Common reward functions for verifiers integration.

Provides reusable reward function implementations that can be
combined and customized for different tasks.
"""

from __future__ import annotations

import json
import re
from abc import ABC
from abc import abstractmethod
from typing import Any

import verifiers as vf
from loguru import logger

from radiant_harness.utils.iou import compute_iou


def extract_completion_text(completion: Any) -> str:
    """Extract text content from a verifiers completion.

    Handles multiple formats:
    - Plain string
    - Message list with assistant role
    - Multimodal content lists

    Args:
        completion: Model completion in any supported format

    Returns:
        Extracted text content
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        # Find last assistant message
        for msg in reversed(completion):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                return content

            # Handle multimodal content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")

    return str(completion or "")


class BaseRewardFunction(ABC):
    """Base class for reward functions.

    Provides a common interface for reward functions that can be
    used with the verifiers package.
    """

    @abstractmethod
    def __call__(
        self,
        prompt: str,
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        """Compute reward for a completion.

        Args:
            prompt: The input prompt
            completion: Model completion
            info: Additional information (e.g., ground truth)

        Returns:
            Reward value (typically 0.0 to 1.0)
        """
        raise NotImplementedError


class ExactMatchReward(BaseRewardFunction):
    """Exact match reward function.

    Rewards 1.0 for exact match, 0.0 otherwise.
    Supports normalization to handle common variations.
    """

    def __init__(
        self,
        normalize: bool = True,
        case_sensitive: bool = False,
        strip_braces: bool = True,
    ) -> None:
        """Initialize exact match reward.

        Args:
            normalize: Whether to normalize strings (lowercase, strip)
            case_sensitive: If False, compare case-insensitively
            strip_braces: Whether to strip braces and punctuation
        """
        self.normalize = normalize
        self.case_sensitive = case_sensitive
        self.strip_braces = strip_braces

    def __call__(
        self,
        prompt: str,  # noqa: ARG002 - Required by interface
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        """Compute exact match reward."""
        pred = extract_completion_text(completion)
        ref = info.get("gold", info.get("reference", info.get("answer", "")))

        if self.normalize:
            pred = self._normalize(pred)
            ref = self._normalize(ref)

        match = pred == ref if self.case_sensitive else pred.lower() == ref.lower()
        return 1.0 if match else 0.0

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""

        if self.strip_braces:
            text = text.strip("{}().[],;")
            text = re.sub(r"\s+", " ", text).strip()

        return text


class TokenF1Reward(BaseRewardFunction):
    """Token-level F1 reward function.

    Computes token overlap between prediction and reference.
    Useful for evaluating text generation where exact match is too strict.
    """

    def __init__(
        self,
        normalize: bool = True,
        case_sensitive: bool = False,
        tokenize: str = "simple",  # "simple", "word", or "character"
    ) -> None:
        """Initialize token F1 reward.

        Args:
            normalize: Whether to normalize strings
            case_sensitive: Whether comparison is case-sensitive
            tokenize: Tokenization method
        """
        self.normalize = normalize
        self.case_sensitive = case_sensitive
        self.tokenize = tokenize

    def __call__(
        self,
        prompt: str,  # noqa: ARG002 - Required by interface
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        """Compute token F1 reward."""
        pred = extract_completion_text(completion)
        ref = info.get("gold", info.get("reference", info.get("answer", "")))

        pred_tokens = self._tokenize_text(pred)
        ref_tokens = self._tokenize_text(ref)

        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        # Count token occurrences
        pred_counts: dict[str, int] = {}
        ref_counts: dict[str, int] = {}

        for token in pred_tokens:
            pred_counts[token] = pred_counts.get(token, 0) + 1
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1

        # Compute intersection
        intersection = sum(
            min(count, ref_counts.get(token, 0)) for token, count in pred_counts.items()
        )

        precision = intersection / len(pred_tokens)
        recall = intersection / len(ref_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def _tokenize_text(self, text: str) -> list[str]:
        """Tokenize text."""
        if not text:
            return []

        # Normalize
        if self.normalize:
            text = text.lower() if not self.case_sensitive else text
            text = re.sub(r"\s+", " ", text.strip())

        # Tokenize
        if self.tokenize == "simple":
            # Split on whitespace and punctuation
            tokens = re.findall(r"\b\w+\b", text)
        elif self.tokenize == "word":
            tokens = text.split()
        elif self.tokenize == "character":
            tokens = list(text)
        else:
            raise ValueError(f"Unknown tokenize method: {self.tokenize}")

        return tokens


class IoUReward(BaseRewardFunction):
    """Intersection over Union (IoU) reward for bounding boxes.

    Rewards based on spatial overlap between predicted and reference boxes.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        normalized: bool = True,
    ) -> None:
        """Initialize IoU reward.

        Args:
            iou_threshold: Minimum IoU for reward
            normalized: Whether coordinates are normalized [0,1]
        """
        self.iou_threshold = iou_threshold
        self.normalized = normalized

    def __call__(
        self,
        prompt: str,  # noqa: ARG002 - Required by interface
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        """Compute IoU reward."""
        pred_box = self._extract_bbox(completion)
        ref_box = info.get("bbox", info.get("reference_bbox", []))

        if not pred_box or not ref_box or len(pred_box) < 4 or len(ref_box) < 4:
            return 0.0

        # Convert to float for compute_iou (handles int coords from JSON)
        pred_floats = [float(x) for x in pred_box[:4]]
        ref_floats = [float(x) for x in ref_box[:4]]
        iou = compute_iou(pred_floats, ref_floats)
        return 1.0 if iou >= self.iou_threshold else iou

    def _extract_bbox(self, completion: Any) -> list[float]:
        """Extract bounding box from completion."""
        text = extract_completion_text(completion)

        # Try to find JSON with bbox
        start = text.find("{")
        if start != -1:
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        json_candidate = text[start : i + 1]
                        try:
                            data = json.loads(json_candidate)
                            if "bbox" in data:
                                return data["bbox"]
                            if "location" in data and "bbox" in data["location"]:
                                return data["location"]["bbox"]
                        except json.JSONDecodeError as e:
                            logger.debug(
                                f"IoUReward: JSON parse failed for bbox extraction: {e}. "
                                f"Snippet: {json_candidate[:100]}..."
                            )
                        break

        # Fallback: regex for [x1, y1, x2, y2] pattern
        pattern = r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
        match = re.search(pattern, text)
        if match:
            return [float(x) for x in match.groups()]

        logger.debug(f"IoUReward: No bbox found in completion. Text length: {len(text)}")
        return []


class CombinedReward(BaseRewardFunction):
    """Combine multiple reward functions with weights."""

    def __init__(
        self,
        rewards: list[BaseRewardFunction],
        weights: list[float] | None = None,
        names: list[str] | None = None,
    ) -> None:
        """Initialize combined reward.

        Args:
            rewards: List of reward functions
            weights: List of weights (must sum to 1.0)
            names: Optional names for each reward
        """
        if not rewards:
            raise ValueError("At least one reward function required")

        self.rewards = rewards
        self.weights = weights or [1.0 / len(rewards)] * len(rewards)
        self.names = names or [f"reward_{i}" for i in range(len(rewards))]

        if len(self.weights) != len(self.rewards):
            raise ValueError("Number of weights must match number of rewards")

        total = sum(self.weights)
        if abs(total - 1.0) > 1e-6:
            # Normalize weights
            self.weights = [w / total for w in self.weights]

    def __call__(
        self,
        prompt: str,
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        """Compute combined reward.

        Note: Stores reward details in info["_reward_details"] for logging.
        This mutation is intentional to allow callers to inspect per-reward scores.
        """
        total_reward = 0.0
        details: dict[str, float] = {}

        for reward, weight, name in zip(self.rewards, self.weights, self.names, strict=True):
            r = reward(prompt, completion, info)
            total_reward += weight * r
            details[name] = r

        # Store details in info for logging (intentional mutation for debugging/logging)
        info["_reward_details"] = details

        return total_reward

    def get_rubric(self) -> Any:
        """Get verifiers rubric for this reward."""
        return vf.Rubric(funcs=self.rewards, weights=self.weights)
