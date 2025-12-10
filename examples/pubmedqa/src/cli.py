"""PubmedQA CLI - demonstrates text-only agentic analysis.

Usage:
    python -m examples.pubmedqa.src.cli --model openai/gpt-4o --use-search
    python -m examples.pubmedqa.src.cli --config pqa_labeled --max-samples 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger

from radiant_harness import AgenticResult

from .dataset import PubmedQADataset
from .evaluation import evaluate_pubmedqa
from .processor import PubmedQAProcessor


async def run_evaluation(
    model_name: str,
    config: str,
    max_samples: int | None,
    use_search: bool,
    max_turns: int,
    output_dir: Path,
    reasoning: bool,
) -> dict[str, object]:
    """Run PubmedQA evaluation.

    Args:
        model_name: Model to use
        config: Dataset configuration
        max_samples: Maximum samples to evaluate
        use_search: Enable web search
        max_turns: Maximum conversation turns
        output_dir: Output directory for results
        reasoning: Enable reasoning mode

    Returns:
        Evaluation metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading PubmedQA dataset (config={config})")
    dataset = PubmedQADataset(config=config, max_samples=max_samples)
    logger.info(f"Loaded {len(dataset)} samples")

    # Create processor
    processor = PubmedQAProcessor(
        model_name=model_name,
        use_web_search=use_search,
        max_turns=max_turns,
        reasoning_enabled=reasoning,
    )

    logger.info(f"Running evaluation with model: {model_name}")
    logger.info(f"Web search: {use_search}, Max turns: {max_turns}")

    # Process samples
    results: list[AgenticResult] = []
    predictions: list[str] = []
    references: list[str] = []

    for i, sample in enumerate(dataset):
        logger.info(f"Processing sample {i + 1}/{len(dataset)}")

        metadata = {
            "question": sample["question"],
            "context": sample["context"],
            "labels": sample["labels"],
            "meshes": sample["meshes"],
            "pubid": sample["pubid"],
        }

        try:
            result = await processor.analyze(images=None, metadata=metadata)
            results.append(result)

            pred_answer = result.final_response.get("answer", "maybe")
            predictions.append(pred_answer)
            references.append(sample["answer"])

            # Save individual result
            result_file = output_dir / f"sample_{i}.json"
            with result_file.open("w") as f:
                json.dump(
                    {
                        "sample_id": i,
                        "pubid": sample["pubid"],
                        "question": sample["question"],
                        "prediction": pred_answer,
                        "ground_truth": sample["answer"],
                        "response": result.final_response,
                        "num_turns": result.num_turns,
                        "tools_used": list(result.get_tools_used()),
                        "confidence": result.confidence,
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Sample {i}: pred={pred_answer}, gt={sample['answer']}, "
                f"turns={result.num_turns}, confidence={result.confidence:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to process sample {i}: {e}")
            predictions.append("maybe")  # Default for failed samples
            references.append(sample["answer"])

    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = evaluate_pubmedqa(predictions, references)

    # Save summary
    summary_file = output_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(
            {
                "config": {
                    "model": model_name,
                    "dataset_config": config,
                    "use_search": use_search,
                    "max_turns": max_turns,
                },
                "num_samples": len(results),
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
        description="PubmedQA Benchmark Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model name (OpenRouter format)",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["pqa_labeled", "pqa_artificial", "pqa_unlabeled"],
        default="pqa_labeled",
        help="Dataset configuration",
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
        default=Path("./runs/pubmedqa"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--use-search",
        action="store_true",
        help="Enable PubMed search for additional context",
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
        logger.enable("examples.pubmedqa")
        logger.enable("radiant_harness")
    else:
        logger.disable("examples.pubmedqa")

    # Run evaluation
    try:
        metrics = asyncio.run(
            run_evaluation(
                model_name=args.model,
                config=args.config,
                max_samples=args.max_samples,
                use_search=args.use_search,
                max_turns=args.max_turns,
                output_dir=args.output_dir,
                reasoning=args.reasoning,
            )
        )
        print("\n=== PubmedQA Results ===")  # noqa: T201
        print(f"Accuracy: {metrics['accuracy']:.3f}")  # noqa: T201
        print(f"Macro F1: {metrics['macro_f1']:.3f}")  # noqa: T201
        print(f"  Yes F1: {metrics['f1_yes']:.3f}")  # noqa: T201
        print(f"  No F1:  {metrics['f1_no']:.3f}")  # noqa: T201
        print(f"  Maybe F1: {metrics['f1_maybe']:.3f}")  # noqa: T201
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
