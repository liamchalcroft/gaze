#!/usr/bin/env python
"""Evaluate GEMeX-ThinkVG with the live processor and reward function."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from loguru import logger

from examples.gemex_thinkvg.src import RewardWeights
from examples.gemex_thinkvg.src import compute_combined_reward
from examples.gemex_thinkvg.src.processor import GEMeXProcessor
from examples.gemex_thinkvg.src.schemas import parse_thinkvg_response
from gaze import require_lmstudio_model
from gaze._frozen import deep_thaw

_QUESTION_TYPE_MAP: dict[str, str] = {
    "open_ended_questions": "open_ended",
    "closed_ended_questions": "closed_ended",
    "single_choice_questions": "single_choice",
    "multi_choice_questions": "multi_choice",
}


class _SafeEncoder(json.JSONEncoder):
    """Handle MappingProxyType from gaze frozen containers."""

    def default(self, o: object) -> Any:
        if isinstance(o, (MappingProxyType, Mapping)):  # noqa: UP038
            return dict(o)
        return super().default(o)


def _load_cases(dataset_path: str) -> list[dict[str, Any]]:
    """Load cases from a JSONL dataset file."""
    cases: list[dict[str, Any]] = []
    with open(dataset_path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                cases.append(json.loads(stripped))
    return cases


def _normalize_question_type(raw_value: str) -> str:
    """Normalize raw GEMeX question-type labels."""
    return _QUESTION_TYPE_MAP.get(raw_value, raw_value or "open_ended")


def _resolve_image_path(raw_path: str, image_dir: Path | None) -> Path:
    """Resolve a GEMeX image path relative to the provided image root."""
    if not raw_path:
        raise FileNotFoundError("Case is missing image_path")

    image_path = Path(raw_path)
    if image_path.is_absolute():
        if image_path.exists():
            return image_path
        if image_dir is None:
            raise FileNotFoundError(
                f"Absolute image_path does not exist: {image_path}. "
                "Pass --image-dir if the dataset stores MIMIC-style leading-slash paths."
            )
        image_path = image_dir / raw_path.lstrip("/")
    elif image_dir is not None:
        image_path = image_dir / raw_path.lstrip("/")

    if not image_path.exists():
        raise FileNotFoundError(
            f"Resolved image_path does not exist: {image_path}. "
            "Pass the MIMIC-CXR image root with --image-dir."
        )
    return image_path


def _reference_from_case(case: dict[str, Any]) -> dict[str, Any]:
    """Build the canonical GEMeX reference payload for scoring."""
    parsed = parse_thinkvg_response(str(case.get("response", ""))) or {}
    parsed_location = parsed.get("location", {})
    bbox = case.get("bbox") or parsed_location.get("bbox") or [0, 0, 0, 0]
    return {
        "answer": case.get("answer") or parsed.get("answer", ""),
        "location": {
            "reference": (
                case.get("location_reference")
                or case.get("location_ref")
                or parsed_location.get("reference", "")
            ),
            "bbox": bbox,
        },
        "question_type": _normalize_question_type(str(case.get("question_type", "open_ended"))),
    }


def _aggregate_metrics(sample_results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate detailed GEMeX reward outputs into benchmark metrics."""
    if not sample_results:
        return {
            "mean_reward": 0.0,
            "answer_accuracy": 0.0,
            "answer_reward": 0.0,
            "location_reward": 0.0,
            "bbox_reward": 0.0,
            "mean_iou": 0.0,
            "iou_50": 0.0,
            "iou_30": 0.0,
            "mean_turns": 0.0,
            "mean_tokens": 0.0,
        }

    n = len(sample_results)

    def _mean(key: str) -> float:
        return sum(float(sample[key]) for sample in sample_results) / n

    return {
        "mean_reward": _mean("reward"),
        "answer_accuracy": _mean("answer_exact_match"),
        "answer_reward": _mean("answer_reward"),
        "location_reward": _mean("location_reward"),
        "bbox_reward": _mean("bbox_reward"),
        "mean_iou": _mean("iou"),
        "iou_50": _mean("iou_50"),
        "iou_30": _mean("iou_30"),
        "mean_turns": _mean("num_turns"),
        "mean_tokens": _mean("total_tokens"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GEMeX-ThinkVG with GAZE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to GEMeX JSONL dataset")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Root directory for resolving image_path entries",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (OpenAI/OpenRouter format, or local ID for --base-url)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible server (e.g. http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single_turn", "agentic"],
        default="agentic",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max turns (single_turn requires 1; agentic defaults to 8)",
    )
    parser.add_argument(
        "--use-tools",
        action="store_true",
        help="Enable visual tools in agentic mode",
    )
    parser.add_argument(
        "--use-web-search",
        action="store_true",
        help="Enable search tools in agentic mode",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--reward-weights",
        type=str,
        default="0.4,0.3,0.3",
        help="Comma-separated weights for answer,location,bbox rewards",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max completion tokens per turn (default: harness default)",
    )
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=None,
        help="Downscale images so neither side exceeds this many pixels before encoding",
    )
    parser.add_argument("--reasoning", action="store_true", help="Enable model reasoning mode")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    return parser.parse_args()


def _resolve_mode(
    mode: str,
    max_turns: int | None,
    use_tools: bool,
    use_web_search: bool,
) -> tuple[int, bool, bool]:
    """Normalize CLI mode into concrete processor settings."""
    if mode == "single_turn":
        if use_tools or use_web_search:
            raise ValueError("--use-tools and --use-web-search are only valid with --mode agentic")
        if max_turns not in (None, 1):
            raise ValueError("--mode single_turn requires --max-turns 1")
        return 1, False, False

    resolved_turns = max_turns if max_turns is not None else 8
    if resolved_turns < 2:
        raise ValueError("--mode agentic requires --max-turns >= 2")
    return resolved_turns, use_tools, use_web_search


async def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    weights = [float(x) for x in args.reward_weights.split(",")]
    if len(weights) != 3:
        raise ValueError("--reward-weights must have 3 values")
    reward_weights = RewardWeights(answer=weights[0], location=weights[1], bbox=weights[2])

    resolved_turns, resolved_use_tools, resolved_use_web_search = _resolve_mode(
        args.mode,
        args.max_turns,
        args.use_tools,
        args.use_web_search,
    )

    loaded_models: list[str] | None = None
    if args.base_url is not None:
        loaded_models = await require_lmstudio_model(model_name=args.model, base_url=args.base_url)
        logger.info(f"LM Studio ready at {args.base_url} with models: {loaded_models}")

    cases = _load_cases(args.dataset)
    if args.num_samples > 0:
        cases = cases[: args.num_samples]

    logger.info(f"Loaded {len(cases)} GEMeX cases from {args.dataset}")

    adapter_factory = None
    if args.base_url is not None:
        from gaze.models import LMStudioAdapter

        _url = args.base_url
        _model = args.model

        def adapter_factory() -> LMStudioAdapter:
            return LMStudioAdapter(model_name=_model, base_url=_url)

    processor = GEMeXProcessor(
        model_name=args.model,
        use_tools=resolved_use_tools,
        use_web_search=resolved_use_web_search,
        max_turns=resolved_turns,
        reasoning_enabled=args.reasoning,
        reward_weights=reward_weights,
        adapter_factory=adapter_factory,
        max_encode_dimension=args.max_image_dim,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )

    sample_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    try:
        for idx, case in enumerate(cases):
            logger.info(f"Processing sample {idx + 1}/{len(cases)}")

            metadata = {
                "question": case.get("question", ""),
                "question_type": _normalize_question_type(
                    str(case.get("question_type", "open_ended"))
                ),
                "options": case.get("options", []),
            }

            try:
                image_path = _resolve_image_path(str(case.get("image_path", "")), args.image_dir)
                result = await processor.analyze(images=image_path, metadata=metadata)
                prediction = deep_thaw(result.final_response)
                reward = compute_combined_reward(
                    prediction=prediction,
                    reference=_reference_from_case(case),
                    weights=reward_weights,
                )

                sample_results.append(
                    {
                        "sample_id": idx,
                        "question": metadata["question"],
                        "image_path": str(image_path),
                        "response": prediction,
                        "num_turns": result.num_turns,
                        "tool_call_count": result.tool_call_count,
                        "tools_used": list(result.get_tools_used()),
                        "confidence": result.confidence,
                        "total_tokens": result.total_tokens,
                        **reward,
                    }
                )
            except Exception as exc:
                logger.error(f"Sample {idx} failed: {exc}")
                partial_response = getattr(exc, "partial_response", None)
                failures.append(
                    {
                        "sample_id": idx,
                        "question": metadata["question"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "partial_response": partial_response,
                    }
                )
    finally:
        await processor.aclose()

    metrics = _aggregate_metrics(sample_results)

    results = {
        "config": {
            "model": args.model,
            "base_url": args.base_url,
            "lmstudio_models": loaded_models,
            "mode": args.mode,
            "max_turns": resolved_turns,
            "max_tokens": args.max_tokens,
            "max_image_dim": args.max_image_dim,
            "temperature": 0.0,
            "seed": args.seed,
            "reasoning": args.reasoning,
            "use_tools": resolved_use_tools,
            "use_web_search": resolved_use_web_search,
        },
        "dataset": args.dataset,
        "image_dir": str(args.image_dir) if args.image_dir else None,
        "num_samples_total": len(cases),
        "num_samples_evaluated": len(sample_results),
        "num_failures": len(failures),
        "reward_weights": {
            "answer": reward_weights.answer,
            "location": reward_weights.location,
            "bbox": reward_weights.bbox,
        },
        "metrics": metrics,
        "failures": failures,
        "sample_results": sample_results,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    results_file = args.output / f"gemex_eval_{args.model.replace('/', '_')}.json"
    with results_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, cls=_SafeEncoder)

    logger.info(f"Results saved to {results_file}")
    return results


def main() -> None:
    args = _parse_args()

    if args.seed is not None:
        import random

        random.seed(args.seed)

    if args.verbose:
        logger.enable("gaze")
        logger.enable("examples.gemex_thinkvg")
    else:
        logger.disable("examples.gemex_thinkvg")

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
