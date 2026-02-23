"""VQA-RAD CLI - demonstrates visual question answering.

Usage:
    python -m examples.vqa_rad.src.cli --model openai/gpt-4o --use-tools
    python -m examples.vqa_rad.src.cli --split test --max-samples 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger

from radiant_harness import AgenticResult

from .dataset import VQARadDataset
from .evaluation import evaluate_closed_only
from .evaluation import evaluate_vqa_rad
from .processor import VQARadProcessor


async def run_evaluation(
    model_name: str,
    split: str,
    max_samples: int | None,
    use_tools: bool,
    use_search: bool,
    max_turns: int,
    output_dir: Path,
    reasoning: bool,
) -> dict[str, float]:
    """Run VQA-RAD evaluation.

    Args:
        model_name: Model to use
        split: Dataset split
        max_samples: Maximum samples to evaluate
        use_tools: Enable visual tools
        use_search: Enable web search
        max_turns: Maximum conversation turns
        output_dir: Output directory for results
        reasoning: Enable reasoning mode

    Returns:
        Evaluation metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading VQA-RAD dataset (split={split})")
    dataset = VQARadDataset(split=split, max_samples=max_samples)
    logger.info(f"Loaded {len(dataset)} samples")

    # Create processor
    processor = VQARadProcessor(
        model_name=model_name,
        use_tools=use_tools,
        use_web_search=use_search,
        max_turns=max_turns,
        reasoning_enabled=reasoning,
    )

    logger.info(f"Running evaluation with model: {model_name}")
    logger.info(f"Tools: {use_tools}, Search: {use_search}, Max turns: {max_turns}")

    # Process samples
    results: list[AgenticResult] = []
    predictions: list[str] = []
    references: list[str] = []
    answer_types: list[str] = []
    num_failures = 0

    for i, sample in enumerate(dataset):
        logger.info(f"Processing sample {i + 1}/{len(dataset)}")

        metadata = {
            "question": sample["question"],
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
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Sample {i}: pred='{pred_answer}', gt='{sample['answer']}', "
                f"type={sample['answer_type']}, turns={result.num_turns}"
            )

        except Exception as e:
            logger.error(f"Failed to process sample {i}: {e}")
            num_failures += 1

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
                    "split": split,
                    "use_tools": use_tools,
                    "use_search": use_search,
                    "max_turns": max_turns,
                },
                "num_samples": len(results),
                "num_failures": num_failures,
                "metrics": metrics,
            },
            f,
            indent=2,
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
        help="Model name (OpenRouter format)",
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
        help="Enable visual manipulation tools",
    )
    parser.add_argument(
        "--use-search",
        action="store_true",
        help="Enable medical literature/image search",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum agentic turns",
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
                split=args.split,
                max_samples=args.max_samples,
                use_tools=args.use_tools,
                use_search=args.use_search,
                max_turns=args.max_turns,
                output_dir=args.output_dir,
                reasoning=args.reasoning,
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
