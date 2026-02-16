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
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from loguru import logger
from PIL import Image

from radiant_harness import AgenticResult

from .config import NOVAConfig
from .config import TaskType
from .data import NovaDataset
from .evaluation.caption import evaluate_caption
from .evaluation.detection import clamp_and_validate_box
from .evaluation.detection import evaluate_detection
from .evaluation.detection import rescale_and_clamp_box
from .evaluation.diagnosis import evaluate_diagnosis_nova_official
from .processor import NOVAAgenticProcessor


class _SafeEncoder(json.JSONEncoder):
    """Handle MappingProxyType and other frozen containers from radiant_harness."""

    def default(self, o: object) -> Any:
        if isinstance(o, (MappingProxyType, Mapping)):  # noqa: UP038
            return dict(o)
        return super().default(o)


async def run_evaluation(config: NOVAConfig) -> dict[str, object]:
    """Run NOVA benchmark evaluation.

    Args:
        config: NOVA configuration

    Returns:
        Dictionary of evaluation results
    """
    # Load NOVA dataset (images + ground truth from HuggingFace by default)
    gt_dir_str = str(config.ground_truth_dir) if config.ground_truth_dir else None
    data_dir_str = str(config.data_dir) if config.data_dir else None
    logger.info(f"Loading NOVA dataset (data_dir={data_dir_str}, gt_dir={gt_dir_str})")
    dataset = NovaDataset(
        data_dir=data_dir_str,
        ground_truth_dir=gt_dir_str,
    )

    # Build adapter factory for custom base URLs (e.g. LM Studio)
    adapter_factory = None
    if config.base_url is not None:
        from radiant_harness.models import LMStudioAdapter

        _url = config.base_url
        _model = config.model_name

        def adapter_factory() -> LMStudioAdapter:
            return LMStudioAdapter(model_name=_model, base_url=_url)

    # Create NOVA processor using radiant_harness
    processor = NOVAAgenticProcessor(
        model_name=config.model_name,
        use_tools=config.use_tools,
        use_web_search=config.use_web_search,
        max_turns=config.max_turns,
        reasoning_enabled=config.reasoning_enabled,
        reasoning_effort=config.reasoning_effort,
        mode=config.mode,
        adapter_factory=adapter_factory,
    )

    logger.info(f"Running NOVA evaluation with model: {config.model_name}")
    logger.info(
        "Task: {}, Tools: {}, Max turns: {}",
        config.task.value,
        config.use_tools,
        config.max_turns,
    )

    # Process samples concurrently using batch_size as concurrency limit
    semaphore = asyncio.Semaphore(config.batch_size)
    results: list[AgenticResult] = []
    ground_truth: list[dict[str, Any]] = []

    # Pre-validate and collect work items; load cached results for skipped samples.
    # max_samples caps the TOTAL samples considered (cached + new).
    sample_limit = config.max_samples if config.max_samples > 0 else len(dataset)
    work_items: list[tuple[int, dict[str, Any]]] = []
    cached_count = 0
    for i, sample in enumerate(dataset):
        if cached_count + len(work_items) >= sample_limit:
            break
        if config.skip_existing:
            result_file = config.output_dir / f"sample_{i}.json"
            if result_file.exists():
                # Load cached result so metrics cover all completed samples
                try:
                    with result_file.open() as f:
                        cached = json.load(f)
                    results.append(
                        AgenticResult(
                            final_response=cached["response"],
                            turns=(),
                            total_tokens=cached.get("total_tokens", 0),
                            confidence=cached.get("confidence", 0.0),
                        )
                    )
                    gt = dict(sample["ground_truth"])
                    image = sample["image"]
                    if isinstance(image, Image.Image):
                        gt["image_width"] = image.width
                        gt["image_height"] = image.height
                    ground_truth.append(gt)
                    cached_count += 1
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning(f"Corrupt cached result {result_file}, re-processing: {exc}")
                    # Fall through to re-process this sample
                else:
                    continue
        if "ground_truth" not in sample:
            raise KeyError(f"Sample {i} missing ground truth data")
        image = sample["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Sample {i} image must be a PIL Image, got {type(image).__name__}")
        work_items.append((i, sample))

    if cached_count:
        print(f"  Loaded {cached_count} cached results from previous run")  # noqa: T201

    async def _process_sample(
        idx: int, sample: dict[str, Any]
    ) -> tuple[int, AgenticResult, dict[str, Any]]:
        async with semaphore:
            logger.info(f"Processing sample {idx + 1}/{len(dataset)}")
            image = sample["image"]
            metadata = dict(sample.get("metadata", {}))
            metadata.setdefault("clinical_history", "")
            metadata.setdefault("modality", "MRI")
            metadata.setdefault("image_id", f"sample_{idx}")

            # Pass PIL Image directly — avoids temp-file PNG save + re-read
            result = await processor.analyze(
                images=image,
                metadata=metadata,
            )

            # Build detailed tool call log for analysis.
            tool_call_log: list[dict[str, Any]] = []
            for turn in result.turns:
                for tc in turn.tool_calls:
                    tool_call_log.append({"name": tc.name, "arguments": tc.arguments})
                for tr in turn.tool_results:
                    # Capture search queries and key metadata (skip large blobs).
                    meta: dict[str, Any] = {}
                    for k in ("query", "search_type", "modality", "body_part", "results_count"):
                        if k in tr.metadata:
                            meta[k] = tr.metadata[k]
                    tool_call_log.append(
                        {
                            "name": tr.tool_name,
                            "result_description": tr.description,
                            "success": tr.success,
                            **meta,
                        }
                    )

            result_file = config.output_dir / f"sample_{idx}.json"
            with result_file.open("w") as f:
                json.dump(
                    {
                        "sample_id": idx,
                        "response": result.final_response,
                        "num_turns": result.num_turns,
                        "tools_used": list(result.get_tools_used()),
                        "tool_call_log": tool_call_log,
                        "confidence": result.confidence,
                        "total_tokens": result.total_tokens,
                    },
                    f,
                    indent=2,
                    cls=_SafeEncoder,
                )

            logger.info(
                f"Sample {idx}: {result.num_turns} turns, "
                f"{result.tool_call_count} tool calls, "
                f"confidence: {result.confidence:.2f}"
            )

            gt = dict(sample["ground_truth"])
            gt["image_width"] = image.width
            gt["image_height"] = image.height
            return idx, result, gt

    failed_samples: list[tuple[int, str]] = []
    if work_items:
        raw = await asyncio.gather(
            *(_process_sample(idx, sample) for idx, sample in work_items),
            return_exceptions=True,
        )
        # Separate successes from failures
        for item, (idx, _sample) in zip(raw, work_items):
            if isinstance(item, BaseException):
                logger.error(f"Sample {idx} failed: {item}")
                failed_samples.append((idx, str(item)))
            else:
                results.append(item[1])
                ground_truth.append(item[2])

    # Report failures (always print to stdout, not just logger, so errors
    # are visible even without -v)
    if failed_samples:
        print(  # noqa: T201
            f"  {len(failed_samples)}/{len(work_items)} samples failed: "
            f"{[i for i, _ in failed_samples]}"
        )
        # Show first error to help diagnose API/auth issues
        first_idx, first_err = failed_samples[0]
        print(f"  First error (sample {first_idx}): {first_err[:300]}")  # noqa: T201

    # Compute evaluation metrics (empty dict if no results)
    metrics: dict[str, object] = {}
    if results:
        logger.info("Computing evaluation metrics...")
        eval_set = set(config.eval_tasks) if config.eval_tasks else None
        metrics = await compute_metrics(results, ground_truth, config.task, eval_set)
    else:
        logger.warning("All samples failed — skipping metric computation")

    # Save summary (always, even if all samples failed)
    summary_file = config.output_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(
            {
                "config": {
                    "model": config.model_name,
                    "mode": config.mode,
                    "task": config.task.value,
                    "use_tools": config.use_tools,
                    "use_web_search": config.use_web_search,
                    "max_turns": config.max_turns,
                },
                "num_samples": len(results),
                "failed_samples": [{"sample_id": idx, "error": err} for idx, err in failed_samples],
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {config.output_dir}")
    if not results:
        raise ValueError(
            f"All {len(failed_samples)} samples failed. "
            f"First error: {failed_samples[0][1][:200] if failed_samples else 'unknown'}"
        )
    return metrics


async def compute_metrics(
    results: list[AgenticResult],
    ground_truth: list[dict[str, Any]],
    task: TaskType,
    eval_tasks: set[str] | None = None,
) -> dict[str, object]:
    """Compute evaluation metrics for results.

    Args:
        results: List of agentic results
        ground_truth: Ground truth annotations aligned with results
        task: Task type to evaluate
        eval_tasks: If provided, only compute metrics for these tasks
            (e.g. {"caption", "localization"}). Overrides *task* filtering.

    Returns:
        Dictionary of metrics
    """
    if not results:
        raise ValueError("No results generated; cannot compute metrics")

    if len(results) != len(ground_truth):
        raise ValueError(
            f"Results and ground truth length mismatch: {len(results)} vs {len(ground_truth)}"
        )

    # Map task names to enum members for lookup
    _TASK_BY_NAME = {t.value: t for t in TaskType}

    def _should_compute(task_name: str) -> bool:
        if eval_tasks is not None:
            return task_name in eval_tasks
        return task in (TaskType.ALL, _TASK_BY_NAME[task_name])

    metrics: dict[str, object] = {}
    predictions = [r.final_response for r in results]

    if _should_compute("caption"):
        pred_captions = []
        for i, p in enumerate(predictions):
            caption = p.get("caption")
            if caption is None:
                raise KeyError(f"Prediction {i} missing 'caption' field")
            if not isinstance(caption, dict):
                raise TypeError(
                    f"Prediction {i} 'caption' must be dict, got {type(caption).__name__}"
                )
            description = caption.get("description")
            if description is None:
                raise KeyError(f"Prediction {i} 'caption' missing required 'description' field")
            pred_captions.append(description)
        gt_captions = [gt.get("caption", "") for gt in ground_truth]
        metrics["caption"] = evaluate_caption(pred_captions, gt_captions)

    if _should_compute("diagnosis"):
        pred_diagnoses = []
        for i, p in enumerate(predictions):
            diag = p.get("diagnosis")
            if diag is None:
                raise KeyError(f"Prediction {i} missing 'diagnosis' field")
            if not isinstance(diag, dict):
                raise TypeError(
                    f"Prediction {i} 'diagnosis' must be dict per NOVA schema, "
                    f"got {type(diag).__name__}"
                )
            primary = diag.get("primary_diagnosis")
            if primary is None:
                raise KeyError(
                    f"Prediction {i} 'diagnosis' missing required 'primary_diagnosis' field"
                )
            # Build ranked list [primary, diff1, diff2, ...] for top-5 evaluation.
            # evaluate_diagnosis_nova_official already handles lists via isinstance(p, list).
            ranked: list[str] = [primary]
            for dd in diag.get("differential_diagnoses", []):
                if isinstance(dd, dict):
                    name = dd.get("diagnosis")
                    if name:
                        ranked.append(name)
                elif isinstance(dd, str) and dd:
                    ranked.append(dd)
            pred_diagnoses.append(ranked)

        gt_diagnoses = [gt.get("final_diagnosis", "") for gt in ground_truth]
        metrics["diagnosis"] = await evaluate_diagnosis_nova_official(pred_diagnoses, gt_diagnoses)

    if _should_compute("localization"):
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
            img_dims = loc_data.get("image_dimensions", {})
            pred_w = img_dims.get("width", 0)
            pred_h = img_dims.get("height", 0)
            actual_w = gt.get("image_width", pred_w)
            actual_h = gt.get("image_height", pred_h)
            boxes = []
            scores = []
            for j, loc in enumerate(localizations):
                # Schema requires 'bounding_box' - enforce strictly
                bbox = loc.get("bounding_box")
                if bbox is None:
                    raise KeyError(
                        f"Prediction {i} localization {j} missing required 'bounding_box' field"
                    )
                bbox = list(bbox)
                if actual_w > 0 and actual_h > 0:
                    bbox = rescale_and_clamp_box(bbox, actual_w, actual_h)
                boxes.append(bbox)
                scores.append(loc.get("confidence", 1.0))
            pred_boxes.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": [0] * len(boxes),
                }
            )

            # Ground truth format: uses 'bbox' field
            gt_localizations = gt.get("localizations", [])
            gt_box_list = [list(loc["bbox"]) for loc in gt_localizations if "bbox" in loc]
            if actual_w > 0 and actual_h > 0:
                gt_box_list = [clamp_and_validate_box(b, actual_w, actual_h) for b in gt_box_list]
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
        help="Model name (OpenRouter format, or local model ID for --base-url)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible server (e.g. http://host:1234/v1)",
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
        default=None,
        help="Path to local NOVA CSV directory (default: load from HuggingFace)",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help="Path to ground truth CSV directory (defaults to --data-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./runs"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["agentic", "single_turn"],
        default="agentic",
        help="Prompt mode: agentic (multi-turn) or single_turn",
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
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all)",
    )
    parser.add_argument(
        "--eval-tasks",
        type=str,
        nargs="+",
        choices=["caption", "diagnosis", "localization"],
        default=None,
        help="Metrics to compute (default: all matching --task)",
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
        base_url=args.base_url,
        task=TaskType(args.task),
        data_dir=args.data_dir,
        ground_truth_dir=args.ground_truth_dir,
        output_dir=args.output_dir,
        use_tools=args.use_tools,
        use_web_search=args.use_web_search,
        max_turns=args.max_turns,
        reasoning_enabled=args.reasoning,
        mode=args.mode,
        eval_tasks=tuple(args.eval_tasks) if args.eval_tasks else None,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
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
