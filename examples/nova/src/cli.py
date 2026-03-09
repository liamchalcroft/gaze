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
import random
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

# AgenticResult deep-freezes dicts into MappingProxyType.  Use this
# tuple in isinstance() checks so both dict and proxy are accepted.
_DICT_LIKE = (dict, MappingProxyType)

from loguru import logger

from radiant_harness._frozen import deep_thaw
from radiant_harness import AgenticResult
from radiant_harness import require_lmstudio_model

from .config import NOVAConfig
from .config import TaskType
from .evaluation.diagnosis import DEFAULT_SEMANTIC_MATCH_MODEL


class _SafeEncoder(json.JSONEncoder):
    """Handle MappingProxyType and other frozen containers from radiant_harness."""

    def default(self, o: object) -> Any:
        if isinstance(o, (MappingProxyType, Mapping)):  # noqa: UP038
            return dict(o)
        return super().default(o)


def _write_json_file(
    path: Path,
    payload: object,
    encoder_cls: type[json.JSONEncoder] | None = None,
) -> None:
    """Write JSON from a worker thread to keep the event loop responsive."""
    with path.open("w") as f:
        if encoder_cls is None:
            json.dump(payload, f, indent=2)
        else:
            json.dump(payload, f, indent=2, cls=encoder_cls)


def _summary_metrics(metrics: dict[str, object]) -> tuple[dict[str, object], list[dict[str, Any]]]:
    """Keep large diagnosis audit logs out of summary.json hot reads."""
    summary_metrics = dict(metrics)
    judgment_log: list[dict[str, Any]] = []

    diagnosis = metrics.get("diagnosis")
    if isinstance(diagnosis, dict):
        diagnosis_summary = dict(diagnosis)
        raw_log = diagnosis_summary.pop("judgment_log", None)
        if isinstance(raw_log, list):
            judgment_log = raw_log
            diagnosis_summary["judgment_log_file"] = "diagnosis_judgment_log.json"
            diagnosis_summary["judgment_log_entries"] = len(raw_log)
        summary_metrics["diagnosis"] = diagnosis_summary

    return summary_metrics, judgment_log


async def run_evaluation(config: NOVAConfig) -> dict[str, object]:
    """Run NOVA benchmark evaluation.

    Args:
        config: NOVA configuration

    Returns:
        Dictionary of evaluation results
    """
    loaded_models: list[str] | None = None
    # Build adapter factory for custom base URLs (e.g. LM Studio)
    adapter_factory = None
    if config.base_url is not None:
        from radiant_harness.models import LMStudioAdapter

        loaded_models = await require_lmstudio_model(
            model_name=config.model_name,
            base_url=config.base_url,
        )
        logger.info(f"LM Studio ready at {config.base_url} with models: {loaded_models}")

        _url = config.base_url
        _model = config.model_name

        def adapter_factory() -> LMStudioAdapter:
            return LMStudioAdapter(model_name=_model, base_url=_url)

    # Load NOVA dataset (images + ground truth from HuggingFace by default)
    from .data import NovaDataset
    from .processor import NOVAAgenticProcessor

    data_dir_str = str(config.data_dir) if config.data_dir else None
    logger.info(f"Loading NOVA dataset (data_dir={data_dir_str})")
    dataset = NovaDataset(
        data_dir=data_dir_str,
    )

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

    try:
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
        # Uses get_sample_metadata() to avoid loading all images into memory upfront;
        # images are loaded on-demand inside _process_sample instead.
        sample_limit = config.max_samples if config.max_samples > 0 else len(dataset)
        work_items: list[int] = []
        cached_count = 0
        for i in range(len(dataset)):
            if cached_count + len(work_items) >= sample_limit:
                break
            sample_meta = dataset.get_sample_metadata(i)
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
                        gt = dict(sample_meta["ground_truth"])
                        img_w, img_h = sample_meta["image_size"]
                        gt["image_width"] = img_w
                        gt["image_height"] = img_h
                        ground_truth.append(gt)
                        cached_count += 1
                    except (json.JSONDecodeError, KeyError) as exc:
                        logger.warning(f"Corrupt cached result {result_file}, re-processing: {exc}")
                        # Fall through to re-process this sample
                    else:
                        continue
            if "ground_truth" not in sample_meta:
                raise KeyError(f"Sample {i} missing ground truth data")
            work_items.append(i)

        if cached_count:
            print(f"  Loaded {cached_count} cached results from previous run")  # noqa: T201

        async def _process_sample(
            idx: int,
        ) -> tuple[int, AgenticResult, dict[str, Any]]:
            async with semaphore:
                logger.info(f"Processing sample {idx + 1}/{len(dataset)}")
                # Load image on demand — keeps peak memory at O(batch_size)
                # instead of O(N) when all images were pre-loaded.
                sample = await asyncio.to_thread(dataset.__getitem__, idx)
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
                        tool_call_log.append(
                            {
                                "name": tc.name,
                                "arguments": tc.arguments
                                if isinstance(tc.arguments, str)
                                else deep_thaw(tc.arguments),
                            }
                        )
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

                sample_payload = {
                    "sample_id": idx,
                    "response": result.final_response,
                    "num_turns": result.num_turns,
                    "tools_used": list(result.get_tools_used()),
                    "tool_call_log": tool_call_log,
                    "confidence": result.confidence,
                    "total_tokens": result.total_tokens,
                }
                result_file = config.output_dir / f"sample_{idx}.json"
                await asyncio.to_thread(_write_json_file, result_file, sample_payload, _SafeEncoder)

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
                *(_process_sample(idx) for idx in work_items),
                return_exceptions=True,
            )
            # Separate successes from failures
            for item, idx in zip(raw, work_items, strict=True):
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

        # Compute token cost summary stats
        token_counts = [r.total_tokens for r in results]
        token_summary: dict[str, object] = {}
        if token_counts:
            sorted_tokens = sorted(token_counts)
            mid = len(sorted_tokens) // 2
            median = (
                sorted_tokens[mid]
                if len(sorted_tokens) % 2 == 1
                else (sorted_tokens[mid - 1] + sorted_tokens[mid]) / 2
            )
            token_summary = {
                "mean_tokens": sum(token_counts) / len(token_counts),
                "median_tokens": median,
                "max_tokens": max(token_counts),
                "total_tokens": sum(token_counts),
            }

        summary_metrics, diagnosis_judgment_log = _summary_metrics(metrics)
        if diagnosis_judgment_log:
            await asyncio.to_thread(
                _write_json_file,
                config.output_dir / "diagnosis_judgment_log.json",
                diagnosis_judgment_log,
            )

        # Save summary (always, even if all samples failed)
        await asyncio.to_thread(
            _write_json_file,
            config.output_dir / "summary.json",
            {
                "config": {
                    "model": config.model_name,
                    "mode": config.mode,
                    "task": config.task.value,
                    "use_tools": config.use_tools,
                    "use_web_search": config.use_web_search,
                    "max_turns": config.max_turns,
                    "base_url": config.base_url,
                    "lmstudio_models": loaded_models,
                    "diagnosis_judge_model": DEFAULT_SEMANTIC_MATCH_MODEL,
                },
                "num_samples": len(results),
                "failed_samples": [{"sample_id": idx, "error": err} for idx, err in failed_samples],
                "metrics": summary_metrics,
                "token_summary": token_summary,
            },
        )

        logger.info(f"Results saved to {config.output_dir}")
        if not results:
            raise ValueError(
                f"All {len(failed_samples)} samples failed. "
                f"First error: {failed_samples[0][1][:200] if failed_samples else 'unknown'}"
            )
        return metrics
    finally:
        await processor.aclose()


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
    task_by_name = {t.value: t for t in TaskType}

    def _should_compute(task_name: str) -> bool:
        if eval_tasks is not None:
            return task_name in eval_tasks
        return task in (TaskType.ALL, task_by_name[task_name])

    metrics: dict[str, object] = {}
    predictions = [r.final_response for r in results]
    pending_metrics: dict[str, asyncio.Task[object]] = {}

    if _should_compute("caption"):
        from .evaluation.caption import evaluate_caption

        pred_captions = []
        for i, p in enumerate(predictions):
            caption = p.get("caption")
            if caption is None or not isinstance(caption, _DICT_LIKE):
                logger.warning(f"Prediction {i} missing or malformed 'caption' — scoring as empty")
                pred_captions.append("")
                continue
            description = caption.get("description", "")
            parts = [description] if description else []
            if seq := caption.get("sequence_characteristics"):
                parts.append(str(seq))
            if orient := caption.get("orientation"):
                parts.append(str(orient))
            parts.extend(str(f) for f in caption.get("findings", []))
            pred_captions.append(" ".join(parts))
        gt_captions = [gt.get("caption", "") for gt in ground_truth]
        pending_metrics["caption"] = asyncio.create_task(
            asyncio.to_thread(evaluate_caption, pred_captions, gt_captions)
        )

    if _should_compute("diagnosis"):
        from .evaluation.diagnosis import evaluate_diagnosis_nova_official

        pred_diagnoses = []
        for i, p in enumerate(predictions):
            diag = p.get("diagnosis")
            if diag is None or not isinstance(diag, _DICT_LIKE):
                logger.warning(
                    f"Prediction {i} missing or malformed 'diagnosis' — scoring as empty"
                )
                pred_diagnoses.append([""])
                continue
            primary = diag.get("primary_diagnosis", "")
            ranked: list[str] = [primary] if primary else [""]
            for dd in diag.get("differential_diagnoses", []):
                if isinstance(dd, _DICT_LIKE):
                    name = dd.get("diagnosis")
                    if name:
                        ranked.append(name)
                elif isinstance(dd, str) and dd:
                    ranked.append(dd)
            pred_diagnoses.append(ranked)

        gt_diagnoses = []
        for i, gt in enumerate(ground_truth):
            diag = gt.get("final_diagnosis", "")
            if not diag:
                logger.warning("Sample %d has empty ground truth diagnosis", i)
            gt_diagnoses.append(diag)
        pending_metrics["diagnosis"] = asyncio.create_task(
            evaluate_diagnosis_nova_official(pred_diagnoses, gt_diagnoses)
        )

    if _should_compute("localization"):
        from .evaluation.detection import clamp_and_validate_box
        from .evaluation.detection import evaluate_detection
        from .evaluation.detection import rescale_and_clamp_box

        pred_boxes = []
        gt_boxes = []
        for i, (p, gt) in enumerate(zip(predictions, ground_truth, strict=True)):
            loc_data = p.get("localization")
            if loc_data is None or not isinstance(loc_data, _DICT_LIKE):
                logger.warning(
                    f"Prediction {i} missing or malformed 'localization' — scoring as empty"
                )
                pred_boxes.append({"boxes": [], "scores": [], "labels": []})
                gt_localizations = gt.get("localizations", [])
                gt_box_list = [list(loc["bbox"]) for loc in gt_localizations if "bbox" in loc]
                gt_boxes.append(
                    {
                        "boxes": gt_box_list,
                        "scores": [1.0] * len(gt_box_list),
                        "labels": [0] * len(gt_box_list),
                    }
                )
                continue
            localizations = loc_data.get("localizations", [])
            img_dims = loc_data.get("image_dimensions", {})
            pred_w = img_dims.get("width", 0)
            pred_h = img_dims.get("height", 0)
            actual_w = gt.get("image_width", pred_w)
            actual_h = gt.get("image_height", pred_h)
            boxes = []
            scores = []
            for j, loc in enumerate(localizations):
                bbox = loc.get("bounding_box")
                if bbox is None:
                    logger.warning(
                        f"Prediction {i} localization {j} missing 'bounding_box' — skipping"
                    )
                    continue
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

        pending_metrics["localization"] = asyncio.create_task(
            asyncio.to_thread(evaluate_detection, pred_boxes, gt_boxes)
        )

    if pending_metrics:
        computed = await asyncio.gather(*pending_metrics.values())
        for name, value in zip(pending_metrics, computed, strict=True):
            metrics[name] = value

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
        help="Base URL for OpenAI-compatible server (audit endpoint: http://192.168.1.138:1234/v1)",
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
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
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

    if args.seed is not None:
        random.seed(args.seed)

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
