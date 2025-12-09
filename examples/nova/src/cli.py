"""NOVA Benchmark CLI - demonstrates radiant_harness usage.

This CLI shows how to use radiant_harness for the NOVA brain-MRI benchmark.
It's intentionally minimal to serve as a reference implementation.

Usage:
    nova-vlm --model openai/gpt-4o --task all --use-tools
    nova-vlm --model anthropic/claude-3.5-sonnet --task diagnosis
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

from radiant_harness import AgenticResult

from .config import NOVAConfig
from .config import TaskType
from .data import NovaDataset
from .evaluation.caption import evaluate_caption
from .evaluation.detection import evaluate_detection
from .evaluation.diagnosis import evaluate_diagnosis_nova_official
from .processor import NOVAAgenticProcessor


async def run_evaluation(config: NOVAConfig) -> dict[str, object]:
    """Run NOVA benchmark evaluation.

    Args:
        config: NOVA configuration

    Returns:
        Dictionary of evaluation results
    """
    # Load NOVA dataset
    logger.info(f"Loading NOVA dataset from {config.data_dir}")
    dataset = NovaDataset(data_dir=str(config.data_dir))

    # Create NOVA processor using radiant_harness
    processor = NOVAAgenticProcessor(
        model_name=config.model_name,
        use_tools=config.use_tools,
        use_web_search=config.use_web_search,
        max_turns=config.max_turns,
        reasoning_enabled=config.reasoning_enabled,
        reasoning_effort=config.reasoning_effort,
    )

    logger.info(f"Running NOVA evaluation with model: {config.model_name}")
    logger.info(
        "Task: {}, Tools: {}, Max turns: {}",
        config.task.value,
        config.use_tools,
        config.max_turns,
    )

    # Process samples
    results: list[AgenticResult] = []
    ground_truth: list[dict[str, Any]] = []
    for i, sample in enumerate(dataset):
        if config.skip_existing:
            result_file = config.output_dir / f"sample_{i}.json"
            if result_file.exists():
                logger.debug(f"Skipping existing result: {result_file}")
                continue

        logger.info(f"Processing sample {i + 1}/{len(dataset)}")

        if "ground_truth" not in sample:
            raise KeyError(f"Sample {i} missing ground truth data")

        # Get image and metadata from dataset
        image = sample["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Sample {i} image must be a PIL Image, got {type(image).__name__}")
        metadata = dict(sample.get("metadata", {}))
        metadata.setdefault("clinical_history", metadata.get("clinical_history", ""))
        metadata.setdefault("modality", metadata.get("modality", "MRI"))
        metadata.setdefault("image_id", metadata.get("image_id", f"sample_{i}"))

        # Save lossless copy temporarily for processing (avoid JPEG artifacts)
        temp_image_path = config.output_dir / f"temp_{i}.png"
        image.save(temp_image_path, format="PNG", optimize=True)

        try:
            # Run analysis using radiant_harness
            result = await processor.analyze(
                images=temp_image_path,
                metadata=metadata,
            )
            results.append(result)
            ground_truth.append(sample["ground_truth"])

            # Save individual result
            result_file = config.output_dir / f"sample_{i}.json"
            with result_file.open("w") as f:
                json.dump(
                    {
                        "sample_id": i,
                        "response": result.final_response,
                        "num_turns": result.num_turns,
                        "tools_used": list(result.get_tools_used()),
                        "confidence": result.confidence,
                        "total_tokens": result.total_tokens,
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Sample {i}: {result.num_turns} turns, "
                f"{result.tool_call_count} tool calls, "
                f"confidence: {result.confidence:.2f}"
            )

        finally:
            # Clean up temp image
            temp_image_path.unlink(missing_ok=True)

    # Compute evaluation metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_metrics(results, ground_truth, config.task)

    # Save summary
    summary_file = config.output_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(
            {
                "config": {
                    "model": config.model_name,
                    "task": config.task.value,
                    "use_tools": config.use_tools,
                    "max_turns": config.max_turns,
                },
                "num_samples": len(results),
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {config.output_dir}")
    return metrics


def compute_metrics(
    results: list[AgenticResult],
    ground_truth: list[dict[str, Any]],
    task: TaskType,
) -> dict[str, object]:
    """Compute evaluation metrics for results.

    Args:
        results: List of agentic results
        ground_truth: Ground truth annotations aligned with results
        task: Task type to evaluate

    Returns:
        Dictionary of metrics
    """
    if not results:
        raise ValueError("No results generated; cannot compute metrics")

    if len(results) != len(ground_truth):
        raise ValueError(
            f"Results and ground truth length mismatch: {len(results)} vs {len(ground_truth)}"
        )

    metrics: dict[str, object] = {}
    predictions = [r.final_response for r in results]

    if task in (TaskType.ALL, TaskType.CAPTION):
        pred_captions = []
        for i, p in enumerate(predictions):
            caption = p.get("caption")
            if caption is None:
                raise KeyError(f"Prediction {i} missing 'caption' field")
            if not isinstance(caption, dict):
                raise TypeError(
                    f"Prediction {i} 'caption' must be dict, got {type(caption).__name__}"
                )
            pred_captions.append(caption.get("description", ""))
        gt_captions = [gt.get("caption", "") for gt in ground_truth]
        metrics["caption"] = evaluate_caption(pred_captions, gt_captions)

    if task in (TaskType.ALL, TaskType.DIAGNOSIS):
        pred_diagnoses = []
        for i, p in enumerate(predictions):
            diag = p.get("diagnosis")
            if diag is None:
                raise KeyError(f"Prediction {i} missing 'diagnosis' field")
            if isinstance(diag, dict):
                pred_diagnoses.append(
                    diag.get("primary_diagnosis") or diag.get("diagnosis") or diag.get("text") or ""
                )
            else:
                pred_diagnoses.append(str(diag))

        gt_diagnoses = [gt.get("final_diagnosis", "") for gt in ground_truth]
        metrics["diagnosis"] = evaluate_diagnosis_nova_official(pred_diagnoses, gt_diagnoses)

    if task in (TaskType.ALL, TaskType.LOCALIZATION):
        pred_boxes = []
        gt_boxes = []
        for i, (p, gt) in enumerate(zip(predictions, ground_truth, strict=True)):
            loc_data = p.get("localization")
            if loc_data is None:
                raise KeyError(f"Prediction {i} missing 'localization' field")
            if not isinstance(loc_data, dict):
                raise TypeError(
                    f"Prediction {i} 'localization' must be dict, got {type(loc_data).__name__}"
                )
            localizations = loc_data.get("localizations", [])
            boxes = []
            scores = []
            for loc in localizations:
                bbox = loc.get("bounding_box") or loc.get("bbox")
                if bbox:
                    boxes.append(bbox)
                    scores.append(loc.get("confidence", 1.0))
            pred_boxes.append(
                {
                    "boxes": boxes,
                    "scores": scores or [1.0] * len(boxes),
                    "labels": [0] * len(boxes),
                }
            )

            gt_localizations = gt.get("localizations", [])
            gt_box_list = []
            for loc in gt_localizations:
                bbox = loc.get("bbox") or loc.get("bounding_box") or loc.get("box")
                if bbox:
                    gt_box_list.append(bbox)
            gt_boxes.append(
                {
                    "boxes": gt_box_list,
                    "scores": [1.0] * len(gt_box_list),
                    "labels": [0] * len(gt_box_list),
                }
            )

        metrics["localization"] = evaluate_detection(pred_boxes, gt_boxes)

    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NOVA Brain-MRI Benchmark Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model name (OpenRouter format)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["all", "caption", "diagnosis", "localization"],
        default="all",
        help="Task to evaluate",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/nova"),
        help="Path to NOVA data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./runs"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--use-tools",
        action="store_true",
        help="Enable visual manipulation tools",
    )
    parser.add_argument(
        "--use-web-search",
        action="store_true",
        help="Enable web search for evidence",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum agentic turns",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable model reasoning mode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess existing results",
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
        logger.enable("src")
        logger.enable("radiant_harness")
    else:
        logger.disable("src")

    # Create configuration
    config = NOVAConfig(
        model_name=args.model,
        task=TaskType(args.task),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_tools=args.use_tools,
        use_web_search=args.use_web_search,
        max_turns=args.max_turns,
        reasoning_enabled=args.reasoning,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing,
    )

    # Run evaluation
    try:
        metrics = asyncio.run(run_evaluation(config))
        print("\n=== NOVA Benchmark Results ===")  # noqa: T201
        print(json.dumps(metrics, indent=2))  # noqa: T201
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
