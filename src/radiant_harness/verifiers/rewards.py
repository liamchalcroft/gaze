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
    Uses continuous IoU values by default to provide smooth gradient signal
    for RL training. A step-function mode is available for binary rewards.

    Includes an optional area penalty to discourage degenerate full-image
    predictions that trivially overlap any ground-truth box.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        normalized: bool = True,
        continuous: bool = True,
        area_penalty_start: float = 0.5,
    ) -> None:
        """Initialize IoU reward.

        Args:
            iou_threshold: IoU threshold used only in step mode
            normalized: Whether coordinates are normalized [0,1]
            continuous: If True (default), return raw IoU for smooth gradients.
                If False, return 1.0 when IoU >= threshold, else 0.0.
            area_penalty_start: Area ratio above which penalty begins.
                When normalized=True, image area is 1.0. A box covering >50%
                of the image starts getting penalized. Set to 1.0 to disable.
        """
        self.iou_threshold = iou_threshold
        self.normalized = normalized
        self.continuous = continuous
        self.area_penalty_start = area_penalty_start

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

        reward = iou if self.continuous else (1.0 if iou >= self.iou_threshold else 0.0)

        # Apply area penalty for degenerate full-image boxes.
        # For normalized coords (in [0,1]), image_area = 1.0.
        # For pixel coords, image_area must be supplied via info dict.
        if self.area_penalty_start < 1.0:
            if self.normalized:
                coords_in_range = all(0.0 <= c <= 1.0 for c in pred_floats)
                if coords_in_range:
                    image_area = 1.0
                else:
                    # Coords are pixel-scale despite normalized=True.
                    # Infer image area from info or max coordinate extent so
                    # the area penalty still applies — prevents reward hacking
                    # by outputting full-image pixel boxes.
                    image_area = float(
                        info.get(
                            "image_area",
                            max(pred_floats[2], pred_floats[0])
                            * max(pred_floats[3], pred_floats[1]),
                        )
                    )
            else:
                image_area = float(info.get("image_area", 0.0))
            if image_area > 0:
                pred_area = abs(pred_floats[2] - pred_floats[0]) * abs(
                    pred_floats[3] - pred_floats[1]
                )
                area_ratio = pred_area / image_area
                if area_ratio > self.area_penalty_start:
                    penalty = max(
                        0.0, (1.0 - area_ratio) / (1.0 - self.area_penalty_start)
                    )
                    reward *= penalty

        return reward

    def _extract_bbox(self, completion: Any) -> list[float]:
        """Extract bounding box from completion.

        Searches all top-level JSON objects in the text, not just the first.
        Models may emit reasoning JSON before the final response that contains
        the bbox, so we need to check every candidate.
        """
        text = extract_completion_text(completion)

        # Search all JSON objects in the text for one containing a bbox
        pos = 0
        while pos < len(text):
            start = text.find("{", pos)
            if start == -1:
                break
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
                            # Also check nested localization structures
                            if "localization" in data:
                                loc = data["localization"]
                                if isinstance(loc, dict):
                                    locs = loc.get("localizations", [])
                                    if isinstance(locs, list) and locs:
                                        first = locs[0]
                                        if isinstance(first, dict):
                                            bbox = first.get(
                                                "bounding_box", first.get("bbox")
                                            )
                                            if isinstance(bbox, list) and len(bbox) >= 4:
                                                return [float(x) for x in bbox[:4]]
                        except json.JSONDecodeError as e:
                            logger.debug(
                                f"IoUReward: JSON parse failed for bbox extraction: {e}. "
                                f"Snippet: {json_candidate[:100]}..."
                            )
                        # Continue searching from after this JSON object
                        pos = i + 1
                        break
            else:
                # Unclosed brace — no more valid JSON possible
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
        """Compute combined reward."""
        total_reward = 0.0
        details: dict[str, float] = {}

        for reward, weight, name in zip(self.rewards, self.weights, self.names, strict=True):
            r = reward(prompt, completion, info)
            total_reward += weight * r
            details[name] = r

        logger.debug(f"CombinedReward details: {details}, total={total_reward:.4f}")

        return total_reward
