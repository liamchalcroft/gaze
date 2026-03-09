"""VQA-RAD CLI - demonstrates visual question answering.

Usage:
    python -m examples.vqa_rad.src.cli --model openai/gpt-4o --use-tools
    python -m examples.vqa_rad.src.cli --split test --max-samples 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from loguru import logger


class _SafeEncoder(json.JSONEncoder):
    """Handle MappingProxyType from radiant_harness frozen containers."""

    def default(self, o: object) -> Any:
        if isinstance(o, (MappingProxyType, Mapping)):  # noqa: UP038
            return dict(o)
        return super().default(o)


from radiant_harness import AgenticResult
from radiant_harness import require_lmstudio_model

from .dataset import VQARadDataset
from .evaluation import evaluate_closed_only
from .evaluation import evaluate_vqa_rad
from .processor import VQARadProcessor


def _resolve_mode(
    mode: str,
    max_turns: int | None,
    use_tools: bool,
    use_search: bool,
) -> tuple[int, bool, bool]:
    """Normalize CLI mode into concrete processor settings."""
    if mode == "single_turn":
        if use_tools or use_search:
            raise ValueError("--use-tools and --use-search are only valid with --mode agentic")
        if max_turns not in (None, 1):
            raise ValueError("--mode single_turn requires --max-turns 1")
        return 1, False, False

    resolved_turns = max_turns if max_turns is not None else 5
    if resolved_turns < 2:
        raise ValueError("--mode agentic requires --max-turns >= 2")
    return resolved_turns, use_tools, use_search


def _failure_record(sample_id: int, question: str, exc: Exception) -> dict[str, object]:
    """Build a stable failure payload for summaries."""
    partial_response = getattr(exc, "partial_response", None)
    return {
        "sample_id": sample_id,
        "question": question,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "partial_response": partial_response,
    }


async def run_evaluation(
    model_name: str,
    mode: str,
    split: str,
    max_samples: int | None,
    use_tools: bool,
    use_search: bool,
    max_turns: int | None,
    output_dir: Path,
    reasoning: bool,
    base_url: str | None = None,
) -> dict[str, float]:
    """Run VQA-RAD evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_max_turns, resolved_use_tools, resolved_use_search = _resolve_mode(
        mode,
        max_turns,
        use_tools,
        use_search,
    )

    loaded_models: list[str] | None = None
    if base_url is not None:
        loaded_models = await require_lmstudio_model(model_name=model_name, base_url=base_url)
        logger.info(f"LM Studio ready at {base_url} with models: {loaded_models}")

    # Load dataset
    logger.info(f"Loading VQA-RAD dataset (split={split})")
    dataset = VQARadDataset(split=split, max_samples=max_samples)
    logger.info(f"Loaded {len(dataset)} samples")

    # Build adapter factory for custom base URLs (e.g. LM Studio)
    adapter_factory = None
    if base_url is not None:
        from radiant_harness.models import LMStudioAdapter

        _url = base_url
        _model = model_name

        def adapter_factory() -> LMStudioAdapter:
            return LMStudioAdapter(model_name=_model, base_url=_url)

    # Create processor
    processor = VQARadProcessor(
        model_name=model_name,
        use_tools=resolved_use_tools,
        use_web_search=resolved_use_search,
        max_turns=resolved_max_turns,
        reasoning_enabled=reasoning,
        adapter_factory=adapter_factory,
    )

    logger.info(f"Running evaluation with model: {model_name}")
    logger.info(
        f"Mode: {mode}, Tools: {resolved_use_tools}, Search: {resolved_use_search}, "
        f"Max turns: {resolved_max_turns}"
    )

    # Process samples
    results: list[AgenticResult] = []
    predictions: list[str] = []
    references: list[str] = []
    answer_types: list[str] = []
    num_failures = 0
    failures: list[dict[str, object]] = []

    try:
        for i, sample in enumerate(dataset):
            logger.info(f"Processing sample {i + 1}/{len(dataset)}")

            metadata = {
                "question": sample["question"],
                "answer_type": sample["answer_type"],
            }

            try:
                result = await processor.analyze(
                    images=sample["image_path"],
                    metadata=metadata,
                )
                results.append(result)

                pred_answer = result.final_response.get("answer", "")
                predictions.append(pred_answer)
                references.append(sample["answer"])
                answer_types.append(sample["answer_type"])

                # Save individual result
                result_file = output_dir / f"sample_{i}.json"
                with result_file.open("w") as f:
                    json.dump(
                        {
                            "sample_id": i,
                            "question": sample["question"],
                            "prediction": pred_answer,
                            "ground_truth": sample["answer"],
                            "answer_type": sample["answer_type"],
                            "response": result.final_response,
                            "num_turns": result.num_turns,
                            "tools_used": list(result.get_tools_used()),
                            "confidence": result.confidence,
                            "total_tokens": result.total_tokens,
                        },
                        f,
                        indent=2,
                        cls=_SafeEncoder,
                    )

                logger.info(
                    f"Sample {i}: pred='{pred_answer}', gt='{sample['answer']}', "
                    f"type={sample['answer_type']}, turns={result.num_turns}"
                )

            except Exception as exc:
                logger.error(f"Failed to process sample {i}: {exc}")
                num_failures += 1
                failures.append(_failure_record(i, sample["question"], exc))
    finally:
        await processor.aclose()

    # Compute metrics (failures are excluded — only successful predictions counted)
    logger.info("Computing evaluation metrics...")
    if not predictions:
        logger.warning("No successful predictions to evaluate")
        metrics: dict[str, float] = {
            "num_failures": float(num_failures),
            "num_samples": 0.0,
        }
    else:
        metrics = evaluate_vqa_rad(predictions, references, answer_types)
        metrics["num_failures"] = float(num_failures)

        # Compute closed-only metrics using the same answer_types used above
        closed_preds = [p for p, t in zip(predictions, answer_types, strict=True) if t == "closed"]
        closed_refs = [r for r, t in zip(references, answer_types, strict=True) if t == "closed"]
        if closed_preds:
            closed_metrics = evaluate_closed_only(closed_preds, closed_refs)
            metrics["closed_binary_accuracy"] = closed_metrics["accuracy"]
            metrics["closed_yes_accuracy"] = closed_metrics["yes_accuracy"]
            metrics["closed_no_accuracy"] = closed_metrics["no_accuracy"]

    # Save summary
    summary_file = output_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(
            {
                "config": {
                    "model": model_name,
                    "mode": mode,
                    "base_url": base_url,
                    "lmstudio_models": loaded_models,
                    "split": split,
                    "use_tools": resolved_use_tools,
                    "use_search": resolved_use_search,
                    "max_turns": resolved_max_turns,
                },
                "num_samples": len(results),
                "num_failures": num_failures,
                "failures": failures,
                "metrics": metrics,
            },
            f,
            indent=2,
            cls=_SafeEncoder,
        )

    logger.info(f"Results saved to {output_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VQA-RAD Benchmark Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model name (OpenRouter format, or local model ID for --base-url)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible server (audit endpoint: http://192.168.1.138:1234/v1)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single_turn", "agentic"],
        default="agentic",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (None for all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./runs/vqa_rad"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--use-tools",
        action="store_true",
        help="Enable visual manipulation tools in agentic mode",
    )
    parser.add_argument(
        "--use-search",
        action="store_true",
        help="Enable medical literature/image search in agentic mode",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max turns (single_turn requires 1; agentic defaults to 5)",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable model reasoning mode",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.verbose:
        logger.enable("examples.vqa_rad")
        logger.enable("radiant_harness")
    else:
        logger.disable("examples.vqa_rad")

    # Run evaluation
    try:
        metrics = asyncio.run(
            run_evaluation(
                model_name=args.model,
                mode=args.mode,
                split=args.split,
                max_samples=args.max_samples,
                use_tools=args.use_tools,
                use_search=args.use_search,
                max_turns=args.max_turns,
                output_dir=args.output_dir,
                reasoning=args.reasoning,
                base_url=args.base_url,
            )
        )
        print("\n=== VQA-RAD Results ===")  # noqa: T201
        if "exact_match" in metrics:
            print(f"Exact Match: {metrics['exact_match']:.3f}")  # noqa: T201
            print(f"Token F1:    {metrics['token_f1']:.3f}")  # noqa: T201
        if "closed_accuracy" in metrics:
            n = int(metrics["num_closed"])
            print(f"Closed Acc:  {metrics['closed_accuracy']:.3f} ({n} samples)")  # noqa: T201
        if "open_accuracy" in metrics:
            n = int(metrics["num_open"])
            print(f"Open Acc:    {metrics['open_accuracy']:.3f} ({n} samples)")  # noqa: T201
            print(f"Open F1:     {metrics['open_f1']:.3f}")  # noqa: T201
        num_failures = int(metrics.get("num_failures", 0))
        if num_failures:
            print(f"Failures:    {num_failures}")  # noqa: T201
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
