#!/usr/bin/env python3
"""
Gather and aggregate evaluation results from benchmark runs.

This script collects metrics from all benchmark runs and generates aggregated results
for analysis and table generation.
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class BenchmarkResults:
    """Container for aggregated benchmark results."""

    approach: str
    task: str
    model: str
    metrics: dict[str, float]
    sample_count: int
    failed_samples: int


def load_individual_metrics(metrics_file: Path) -> dict[str, float] | None:
    """Load metrics from a single metrics.json file."""
    try:
        with open(metrics_file) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load {metrics_file}: {e}")
        return None


def aggregate_task_metrics(task: str, metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate metrics for a specific task."""
    if not metrics_list:
        return {}

    # Task-specific metric aggregation using actual saved metric names
    if task == "localization":
        # For localization, we use detection metrics
        return {
            "iou": np.mean(
                [m.get("detection_mAP50", 0.0) for m in metrics_list]
            ),  # Use mAP50 as IoU proxy
            "map30": np.mean([m.get("detection_mAP30", 0.0) for m in metrics_list]),
            "map50": np.mean([m.get("detection_mAP50", 0.0) for m in metrics_list]),
            "map50_95": np.mean([m.get("detection_mAP50_95", 0.0) for m in metrics_list]),
        }
    elif task == "caption":
        # For caption, we use caption-prefixed metrics
        return {
            "bleu": np.mean([m.get("caption_bleu", 0.0) for m in metrics_list]),
            "bleu4": np.mean([m.get("caption_bleu", 0.0) for m in metrics_list]),  # Alias for table
            "bert_f1": np.mean([m.get("caption_bert_f1", 0.0) for m in metrics_list]),
            "meteor": np.mean([m.get("caption_meteor", 0.0) for m in metrics_list]),
            "radgraph_f1": np.mean([m.get("caption_radgraph_f1", 0.0) for m in metrics_list]),
            "modality_f1": np.mean([m.get("caption_modality_f1", 0.0) for m in metrics_list]),
            "clinical_f1": np.mean([m.get("caption_clinical_f1", 0.0) for m in metrics_list]),
            "binary_f1": np.mean([m.get("caption_binary_f1", 0.0) for m in metrics_list]),
        }
    elif task == "diagnosis":
        # For diagnosis, we use official NOVA protocol with GPT-4o semantic matching
        return {
            "accuracy": np.mean([m.get("diagnosis_top1", 0.0) for m in metrics_list]),
            "top1": np.mean([m.get("diagnosis_top1", 0.0) for m in metrics_list]),
            "top5": np.mean([m.get("diagnosis_top5", 0.0) for m in metrics_list]),
            "coverage": np.mean([m.get("diagnosis_coverage", 0.0) for m in metrics_list]),
            "entropy": np.mean([m.get("diagnosis_entropy", 0.0) for m in metrics_list]),
        }
    else:
        # Generic aggregation for unknown tasks
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())

        return {
            str(key): float(np.mean([m.get(key, 0.0) for m in metrics_list]))
            for key in all_keys
            if isinstance(key, str)
        }


def gather_results_from_directory(results_dir: Path) -> list[BenchmarkResults]:
    """Gather all results from a benchmark results directory."""
    results = []

    # Expected structure: results_dir/{approach}/{task}/{model}/{timestamp}/image_{i}/metrics.json
    for approach_dir in results_dir.iterdir():
        if not approach_dir.is_dir():
            continue

        approach = approach_dir.name
        logger.info(f"Processing approach: {approach}")

        for task_dir in approach_dir.iterdir():
            if not task_dir.is_dir():
                continue

            task = task_dir.name
            logger.info(f"  Processing task: {task}")

            for model_dir in task_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model = model_dir.name
                logger.info(f"    Processing model: {model}")

                # Find all metrics files for this approach/task/model combination
                metrics_files = list(model_dir.glob("*/image_*/metrics.json"))

                # Load all metrics
                all_metrics = []
                failed_count = 0

                for metrics_file in metrics_files:
                    metrics = load_individual_metrics(metrics_file)
                    if metrics is not None:
                        all_metrics.append(metrics)
                    else:
                        failed_count += 1

                if all_metrics:
                    # Aggregate metrics for this combination
                    aggregated_metrics = aggregate_task_metrics(task, all_metrics)

                    result = BenchmarkResults(
                        approach=approach,
                        task=task,
                        model=model,
                        metrics=aggregated_metrics,
                        sample_count=len(all_metrics),
                        failed_samples=failed_count,
                    )
                    results.append(result)

                    logger.info(f"      Found {len(all_metrics)} samples, {failed_count} failed")
                else:
                    logger.warning(f"      No valid metrics found for {approach}/{task}/{model}")

    return list(results)


def save_results(results: list[BenchmarkResults], output_dir: Path) -> None:
    """Save aggregated results to various formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to structured data for easy manipulation
    data = []
    for result in results:
        base_row = {
            "approach": result.approach,
            "task": result.task,
            "model": result.model,
            "sample_count": result.sample_count,
            "failed_samples": result.failed_samples,
        }

        # Add all metrics as columns
        row = {**base_row, **result.metrics}
        data.append(row)

    # Save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "aggregated_results.csv", index=False)
    logger.info(f"Saved CSV results to {output_dir / 'aggregated_results.csv'}")

    # Save as JSON for programmatic access
    json_data = {
        "results": [
            {
                "approach": r.approach,
                "task": r.task,
                "model": r.model,
                "metrics": r.metrics,
                "sample_count": r.sample_count,
                "failed_samples": r.failed_samples,
            }
            for r in results
        ],
        "metadata": {
            "total_results": len(results),
            "approaches": list({r.approach for r in results}),
            "tasks": list({r.task for r in results}),
            "models": list({r.model for r in results}),
        },
    }

    with open(output_dir / "aggregated_results.json", "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Saved JSON results to {output_dir / 'aggregated_results.json'}")

    # Create summary by approach and task (averaged across models)
    summary_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for result in results:
        for metric_name, metric_value in result.metrics.items():
            summary_data[result.approach][f"{result.task}_{metric_name}"].append(metric_value)

    # Average across models for each approach/task/metric combination
    summary_rows = []
    for approach, task_metrics in summary_data.items():
        if isinstance(approach, str):
            row = {"approach": approach}
            for task_metric, values in task_metrics.items():
                if isinstance(task_metric, str):
                    row[task_metric] = float(np.mean(values)) if values else 0.0
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary_by_approach.csv", index=False)
    logger.info(f"Saved summary to {output_dir / 'summary_by_approach.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Gather and aggregate benchmark results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="runs/full_benchmark",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/results_analysis",
        help="Directory to save aggregated results",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if not args.verbose:
        logger.remove()
        logger.add(lambda _: None)  # Suppress logs if not verbose

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return 1

    logger.info(f"Gathering results from: {results_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Gather all results
    results = gather_results_from_directory(results_dir)

    if not results:
        logger.error("No results found!")
        return 1

    logger.info(f"Found {len(results)} result combinations")

    # Save results
    save_results(results, output_dir)

    logger.info("Results gathering completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
