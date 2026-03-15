"""PubmedQA CLI - demonstrates text-only agentic analysis.

Usage:
    python -m examples.pubmedqa.src.cli --model openai/gpt-4o --use-search
    python -m examples.pubmedqa.src.cli --config pqa_labeled --max-samples 100
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

from .dataset import PubmedQADataset
from .evaluation import evaluate_pubmedqa
from .processor import PubmedQAProcessor


def _resolve_mode(
    mode: str,
    max_turns: int | None,
    use_search: bool,
) -> tuple[int, bool]:
    """Normalize CLI mode into concrete processor settings."""
    if mode == "single_turn":
        if use_search:
            raise ValueError("--use-search is only valid with --mode agentic")
        if max_turns not in (None, 1):
            raise ValueError("--mode single_turn requires --max-turns 1")
        return 1, False

    resolved_turns = max_turns if max_turns is not None else 5
    if resolved_turns < 2:
        raise ValueError("--mode agentic requires --max-turns >= 2")
    return resolved_turns, use_search


def _failure_record(sample_id: int, pubid: str, exc: Exception) -> dict[str, object]:
    """Build a stable failure payload for summaries."""
    return {
        "sample_id": sample_id,
        "pubid": pubid,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


async def run_evaluation(
    model_name: str,
    mode: str,
    config: str,
    max_samples: int | None,
    use_search: bool,
    max_turns: int | None,
    output_dir: Path,
    reasoning: bool,
    base_url: str | None = None,
    max_tokens: int | None = None,
    batch_size: int = 1,
    seed: int | None = None,
) -> dict[str, float]:
    """Run PubmedQA evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_max_turns, resolved_use_search = _resolve_mode(mode, max_turns, use_search)

    loaded_models: list[str] | None = None
    if base_url is not None:
        loaded_models = await require_lmstudio_model(model_name=model_name, base_url=base_url)
        logger.info(f"LM Studio ready at {base_url} with models: {loaded_models}")

    # Load dataset
    logger.info(f"Loading PubmedQA dataset (config={config})")
    dataset = PubmedQADataset(config=config, max_samples=max_samples)
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
    processor = PubmedQAProcessor(
        model_name=model_name,
        use_web_search=resolved_use_search,
        max_turns=resolved_max_turns,
        reasoning_enabled=reasoning,
        adapter_factory=adapter_factory,
        seed=seed,
        max_tokens=max_tokens,
    )

    logger.info(f"Running evaluation with model: {model_name}")
    logger.info(f"Mode: {mode}, Web search: {resolved_use_search}, Max turns: {resolved_max_turns}")

    # Process samples concurrently using batch_size as concurrency limit
    semaphore = asyncio.Semaphore(batch_size)
    results: list[AgenticResult] = []
    predictions: list[str] = []
    references: list[str] = []
    failures: list[dict[str, object]] = []

    async def _process_sample(
        i: int, sample: dict[str, Any],
    ) -> tuple[int, AgenticResult, str, str]:
        async with semaphore:
            logger.info(f"Processing sample {i + 1}/{len(dataset)}")
            metadata = {
                "question": sample["question"],
                "context": sample["context"],
                "labels": sample["labels"],
                "meshes": sample["meshes"],
                "pubid": sample["pubid"],
            }
            result = await processor.analyze(images=None, metadata=metadata)

            pred_answer = result.final_response.get("answer", "maybe")

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
                        "total_tokens": result.total_tokens,
                    },
                    f,
                    indent=2,
                    cls=_SafeEncoder,
                )

            logger.info(
                f"Sample {i}: pred={pred_answer}, gt={sample['answer']}, "
                f"turns={result.num_turns}, confidence={result.confidence:.2f}"
            )
            return i, result, pred_answer, sample["answer"]

    try:
        raw = await asyncio.gather(
            *(_process_sample(i, sample) for i, sample in enumerate(dataset)),
            return_exceptions=True,
        )
        for idx, (item, sample) in enumerate(zip(raw, dataset, strict=True)):
            if isinstance(item, BaseException):
                logger.error(f"Failed to process sample {idx}: {item}")
                failures.append(_failure_record(
                    idx,
                    sample["pubid"],
                    item if isinstance(item, Exception) else Exception(str(item)),
                ))
            else:
                _, result, pred_answer, ref_answer = item
                results.append(result)
                predictions.append(pred_answer)
                references.append(ref_answer)
    finally:
        await processor.aclose()

    # Report failures
    if failures:
        logger.warning(
            f"{len(failures)} of {len(dataset)} samples failed processing "
            f"and are excluded from metrics"
        )

    # Compute metrics
    if not predictions:
        logger.error("All samples failed — no predictions to evaluate")
        metrics: dict[str, float] = {"accuracy": 0.0, "macro_f1": 0.0}
    else:
        logger.info("Computing evaluation metrics...")
        metrics = evaluate_pubmedqa(predictions, references)

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
                    "dataset_config": config,
                    "use_search": resolved_use_search,
                    "max_turns": resolved_max_turns,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "seed": seed,
                },
                "num_samples_total": len(dataset),
                "num_samples_evaluated": len(predictions),
                "num_failures": len(failures),
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
        description="PubmedQA Benchmark Evaluation",
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
        help="Enable PubMed search for additional context in agentic mode",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max turns (single_turn requires 1; agentic defaults to 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max completion tokens per turn (default: harness default 16384)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples to process concurrently",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable model reasoning mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
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

    if args.seed is not None:
        import random

        random.seed(args.seed)

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
                mode=args.mode,
                config=args.config,
                max_samples=args.max_samples,
                use_search=args.use_search,
                max_turns=args.max_turns,
                output_dir=args.output_dir,
                reasoning=args.reasoning,
                base_url=args.base_url,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        )
        print("\n=== PubmedQA Results ===")  # noqa: T201
        print(f"Accuracy: {metrics['accuracy']:.3f}")  # noqa: T201
        print(f"Macro F1: {metrics['macro_f1']:.3f}")  # noqa: T201
        if "f1_yes" in metrics:
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
