"""Combined reward function for GEMeX-ThinkVG RL training.

Combines answer, location, and bounding box rewards with configurable
weights for verifiable reward computation.

Also provides GEMeXVerifiersReward for integration with radiant_harness verifiers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from beartype import beartype

from radiant_harness.utils import extract_json_from_text
from radiant_harness.verifiers import BaseRewardFunction
from radiant_harness.verifiers import extract_completion_text

from ..schemas import validate_gemex_response
from .answer import compute_answer_reward
from .bbox import IMAGE_SIZE
from .bbox import compute_bbox_reward
from .location import compute_location_reward


@dataclass(frozen=True)
class RewardWeights:
    """Weights for combining reward components."""

    answer: float = 0.4
    location: float = 0.3
    bbox: float = 0.3

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0."""
        total = self.answer + self.location + self.bbox
        if abs(total - 1.0) > 1e-6:
            msg = f"Reward weights must sum to 1.0, got {total}"
            raise ValueError(msg)


# Default weights for GEMeX task
DEFAULT_WEIGHTS = RewardWeights(answer=0.4, location=0.3, bbox=0.3)


@beartype
def compute_combined_reward(
    prediction: dict[str, Any],
    reference: dict[str, Any],
    weights: RewardWeights = DEFAULT_WEIGHTS,
    image_size: int = IMAGE_SIZE,
) -> dict[str, Any]:
    """Compute combined verifiable reward for GEMeX-ThinkVG.

    Combines three verifiable reward components:
    1. Answer reward - semantic matching of medical findings
    2. Location reward - anatomical region matching
    3. BBox reward - IoU-based bounding box accuracy

    Args:
        prediction: Model prediction with keys:
            - answer: str - predicted medical finding
            - location: dict with 'reference' (str) and 'bbox' (list)
        reference: Ground truth with same structure
        weights: Component weights (must sum to 1.0)
        image_size: Image dimension for bbox validation

    Returns:
        Dict with all component scores and combined reward
    """
    # Extract components
    pred_answer = prediction.get("answer", "")
    ref_answer = reference.get("answer", "")

    pred_location = prediction.get("location", {})
    ref_location = reference.get("location", {})

    pred_loc_ref = pred_location.get("reference", "")
    ref_loc_ref = ref_location.get("reference", "")

    pred_bbox = pred_location.get("bbox", [0, 0, 0, 0])
    ref_bbox = ref_location.get("bbox", [0, 0, 0, 0])

    question_type = reference.get("question_type", "open_ended")

    # Compute component rewards
    answer_result = compute_answer_reward(
        prediction=pred_answer,
        reference=ref_answer,
        question_type=question_type,
    )

    location_result = compute_location_reward(
        prediction=pred_loc_ref,
        reference=ref_loc_ref,
    )

    bbox_result = compute_bbox_reward(
        prediction=pred_bbox,
        reference=ref_bbox,
        image_size=image_size,
    )

    # Compute weighted combined reward
    combined_reward = (
        weights.answer * answer_result["reward"]
        + weights.location * location_result["reward"]
        + weights.bbox * bbox_result["reward"]
    )

    return {
        # Combined reward
        "reward": combined_reward,
        # Component rewards
        "answer_reward": answer_result["reward"],
        "location_reward": location_result["reward"],
        "bbox_reward": bbox_result["reward"],
        # Answer details
        "answer_exact_match": answer_result["exact_match"],
        "answer_contains_match": answer_result["contains_match"],
        "answer_token_f1": answer_result["token_f1"],
        # Location details
        "location_exact_match": location_result["exact_match"],
        "location_hierarchy_match": location_result["hierarchy_match"],
        "location_token_overlap": location_result["token_overlap"],
        "pred_canonical_region": location_result["pred_canonical"],
        "ref_canonical_region": location_result["ref_canonical"],
        # BBox details
        "iou": bbox_result["iou"],
        "giou": bbox_result["giou"],
        "center_distance": bbox_result["center_distance"],
        "iou_50": bbox_result["iou_50"],
        "iou_30": bbox_result["iou_30"],
        "valid_prediction": bbox_result["valid_prediction"],
        # Weights used
        "weights": {
            "answer": weights.answer,
            "location": weights.location,
            "bbox": weights.bbox,
        },
    }


@beartype
def compute_batch_combined_rewards(
    predictions: Sequence[dict[str, Any]],
    references: Sequence[dict[str, Any]],
    weights: RewardWeights = DEFAULT_WEIGHTS,
    image_size: int = IMAGE_SIZE,
) -> list[dict[str, Any]]:
    """Compute combined rewards for a batch of samples.

    Args:
        predictions: List of model predictions
        references: List of ground truth references
        weights: Component weights
        image_size: Image dimension

    Returns:
        List of reward dicts for each sample
    """
    return [
        compute_combined_reward(pred, ref, weights, image_size)
        for pred, ref in zip(predictions, references, strict=True)
    ]


class GEMeXRewardFunction:
    """Callable reward function for veRL integration.

    Wraps compute_combined_reward for use as a verifiable reward
    function in reinforcement learning training.

    Example:
        reward_fn = GEMeXRewardFunction(
            weights=RewardWeights(answer=0.5, location=0.25, bbox=0.25)
        )
        rewards = reward_fn(predictions, references)
    """

    def __init__(
        self,
        weights: RewardWeights | None = None,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        """Initialize reward function.

        Args:
            weights: Optional custom weights (defaults to 0.4/0.3/0.3)
            image_size: Image dimension for bbox validation
        """
        self.weights = weights or DEFAULT_WEIGHTS
        self.image_size = image_size

    @beartype
    def __call__(
        self,
        predictions: Sequence[dict[str, Any]],
        references: Sequence[dict[str, Any]],
    ) -> list[float]:
        """Compute rewards for a batch.

        Args:
            predictions: Model predictions
            references: Ground truth references

        Returns:
            List of scalar rewards for each sample
        """
        results = compute_batch_combined_rewards(
            predictions=predictions,
            references=references,
            weights=self.weights,
            image_size=self.image_size,
        )
        return [r["reward"] for r in results]

    @beartype
    def compute_detailed(
        self,
        predictions: Sequence[dict[str, Any]],
        references: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compute detailed rewards with all component scores.

        Args:
            predictions: Model predictions
            references: Ground truth references

        Returns:
            List of detailed reward dicts
        """
        return compute_batch_combined_rewards(
            predictions=predictions,
            references=references,
            weights=self.weights,
            image_size=self.image_size,
        )

    def get_metrics(
        self,
        results: Sequence[dict[str, Any]],
    ) -> dict[str, float]:
        """Aggregate metrics from detailed results.

        Args:
            results: List of detailed reward dicts

        Returns:
            Aggregated metrics dictionary
        """
        if not results:
            return {}

        n = len(results)

        # Compute means
        metrics = {
            "reward_mean": sum(r["reward"] for r in results) / n,
            "answer_reward_mean": sum(r["answer_reward"] for r in results) / n,
            "location_reward_mean": sum(r["location_reward"] for r in results) / n,
            "bbox_reward_mean": sum(r["bbox_reward"] for r in results) / n,
            "iou_mean": sum(r["iou"] for r in results) / n,
            "iou_50_accuracy": sum(1 for r in results if r["iou_50"]) / n,
            "iou_30_accuracy": sum(1 for r in results if r["iou_30"]) / n,
            "answer_exact_match_rate": sum(r["answer_exact_match"] for r in results)
            / n,
            "location_exact_match_rate": sum(r["location_exact_match"] for r in results)
            / n,
            "valid_prediction_rate": sum(1 for r in results if r["valid_prediction"])
            / n,
        }

        return metrics


class GEMeXVerifiersReward(BaseRewardFunction):
    """Verifiers-compatible reward function for GEMeX-ThinkVG.

    Adapts the GEMeXRewardFunction to the radiant_harness BaseRewardFunction
    interface for use with verifiers training and evaluation.

    Example:
        reward_fn = GEMeXVerifiersReward()
        reward = reward_fn(prompt, completion, info)

        # Or via processor:
        processor = GEMeXProcessor(reward_weights=RewardWeights(0.5, 0.25, 0.25))
        env_cls = processor.as_verifiers_env(dataset_path="train.jsonl")
    """

    def __init__(
        self,
        weights: RewardWeights | None = None,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        """Initialize verifiers reward adapter.

        Args:
            weights: Component weights for answer/location/bbox
            image_size: Image dimension for bbox validation
        """
        self.weights = weights or DEFAULT_WEIGHTS
        self.image_size = image_size
        self._reward_fn = GEMeXRewardFunction(weights=self.weights, image_size=image_size)

    def __call__(
        self,
        prompt: str,  # noqa: ARG002 - Required by interface
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        """Compute GEMeX reward for a completion.

        Args:
            prompt: The input prompt (unused, required by interface)
            completion: Model completion (string or message list)
            info: Case information with ground truth

        Returns:
            Combined reward value in [0.0, 1.0]
        """
        # Extract completion text
        comp_text = self._extract_text(completion)

        # Parse response from completion
        response = self._extract_json_response(comp_text)
        if response is None or not validate_gemex_response(response):
            return 0.0

        # Build prediction dict from response
        prediction = {
            "answer": response.get("answer", ""),
            "location": {
                "reference": response.get("location", {}).get("reference", ""),
                "bbox": response.get("location", {}).get("bbox", [0, 0, 0, 0]),
            },
        }

        # Build reference dict from info
        reference = {
            "answer": info.get("gold_answer", info.get("answer", "")),
            "location": {
                "reference": info.get("gold_location", info.get("location_reference", "")),
                "bbox": info.get("gold_bbox", info.get("bbox", [0, 0, 0, 0])),
            },
            "question_type": info.get("question_type", "open_ended"),
        }

        # Compute reward using existing reward function
        rewards = self._reward_fn([prediction], [reference])
        return rewards[0]

    @staticmethod
    def _extract_text(completion: Any) -> str:
        """Extract text from completion."""
        return extract_completion_text(completion)

    @staticmethod
    def _extract_json_response(text: str) -> dict[str, Any] | None:
        """Extract JSON response from text, with XML fallback."""
        from ..schemas import parse_thinkvg_response

        result = extract_json_from_text(text)
        if result is not None:
            return result
        return parse_thinkvg_response(text)
