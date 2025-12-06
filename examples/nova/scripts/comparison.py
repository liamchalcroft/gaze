#!/usr/bin/env python3
"""NOVA Comparison Script

Aggregates metrics from multiple experimental runs and creates comparison plots and tables.

Usage:
    python scripts/compare.py --parent-dir ./results --output ./paper_results
    python scripts/compare.py --parent-dir ./results --models baseline agentic baseline_reasoning
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype
from loguru import logger


@beartype
def load_metrics_from_runs(
    parent_dir: Path, run_names: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Load evaluation metrics from multiple runs."""
    all_metrics = {}

    # Find all metrics directories
    metrics_dirs = []

    if run_names:
        # Use specified run names
        for run_name in run_names:
            metrics_dir = parent_dir / f"{run_name}_metrics"
            if metrics_dir.exists():
                metrics_dirs.append(metrics_dir)
            else:
                logger.warning(f"Metrics directory not found: {metrics_dir}")
    else:
        # Find all metrics directories automatically
        metrics_dirs.extend(
            [
                item
                for item in parent_dir.iterdir()
                if item.is_dir() and item.name.endswith("_metrics")
            ]
        )

    logger.info(f"Found {len(metrics_dirs)} metrics directories")

    for metrics_dir in metrics_dirs:
        metrics_file = metrics_dir / "evaluation_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                run_name = metrics_dir.name.replace("_metrics", "")
                all_metrics[run_name] = metrics
                logger.info(f"Loaded metrics for run: {run_name}")
        else:
            logger.warning(f"No evaluation_metrics.json found in {metrics_dir}")

    return all_metrics


@beartype
def create_comparison_table(all_metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Create comparison table from all metrics."""
    comparison_data = []

    for run_name, metrics in all_metrics.items():
        overall = metrics.get("overall_summary", {})
        caption = metrics.get("caption_metrics", {})
        diagnosis = metrics.get("diagnosis_metrics", {})
        localization = metrics.get("localization_metrics", {})

        row = {
            "Run": run_name.replace("_", " ").title(),
            "Total Subjects": overall.get("total_subjects", 0),
            "Caption Success": f"{overall.get('tasks_completed', {}).get('caption', 0):.1%}",
            "Diagnosis Success": f"{overall.get('tasks_completed', {}).get('diagnosis', 0):.1%}",
            "Localization Success": (
                f"{overall.get('tasks_completed', {}).get('localization', 0):.1%}"
            ),
            "Avg Confidence": f"{overall.get('average_confidence', 0):.3f}",
            "Avg Caption Length": f"{caption.get('avg_caption_length', 0):.1f}",
            "Diagnosis Categories": len(diagnosis.get("diagnosis_categories", {})),
            "Total Detections": localization.get("total_detections", 0),
            "Detections Per Subject": f"{localization.get('avg_detections_per_subject', 0):.2f}",
        }
        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


@beartype
def create_success_rate_plot(all_metrics: dict[str, dict[str, Any]], output_dir: Path) -> None:
    """Create bar plot comparing success rates across tasks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    runs = list(all_metrics.keys())
    caption_rates = []
    diagnosis_rates = []
    localization_rates = []

    for metrics in all_metrics.values():
        overall = metrics.get("overall_summary", {})
        tasks = overall.get("tasks_completed", {})
        caption_rates.append(tasks.get("caption", 0))
        diagnosis_rates.append(tasks.get("diagnosis", 0))
        localization_rates.append(tasks.get("localization", 0))

    # Success rate bar plot
    x = np.arange(len(runs))
    width = 0.25

    ax1.bar(x - width, caption_rates, width, label="Caption", color="skyblue", alpha=0.8)
    ax1.bar(x, diagnosis_rates, width, label="Diagnosis", color="lightcoral", alpha=0.8)
    ax1.bar(
        x + width, localization_rates, width, label="Localization", color="lightgreen", alpha=0.8
    )

    ax1.set_xlabel("Experimental Runs")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Task Success Rates Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels([run.replace("_", " ").title() for run in runs], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Confidence comparison
    confidences = []
    for metrics in all_metrics.values():
        overall = metrics.get("overall_summary", {})
        confidences.append(overall.get("average_confidence", 0))

    bars = ax2.bar(runs, confidences, color="gold", alpha=0.8)
    ax2.set_xlabel("Experimental Runs")
    ax2.set_ylabel("Average Confidence")
    ax2.set_title("Model Confidence Comparison")
    ax2.set_xticklabels([run.replace("_", " ").title() for run in runs], rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Add value labels on bars
    for bar, conf in zip(bars, confidences, strict=False):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{conf:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plot_file = output_dir / "success_rates_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved success rates plot: {plot_file}")


@beartype
def create_detection_stats_plot(all_metrics: dict[str, dict[str, Any]], output_dir: Path) -> None:
    """Create plot comparing detection statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    runs = list(all_metrics.keys())
    total_detections = []
    detections_per_subject = []
    detection_rates = []

    for metrics in all_metrics.values():
        loc = metrics.get("localization_metrics", {})
        total_detections.append(loc.get("total_detections", 0))
        detections_per_subject.append(loc.get("avg_detections_per_subject", 0))
        detection_rates.append(loc.get("detection_rate", 0))

    # Total detections
    bars1 = ax1.bar(runs, total_detections, color="orange", alpha=0.8)
    ax1.set_xlabel("Experimental Runs")
    ax1.set_ylabel("Total Detections")
    ax1.set_title("Total Detections per Run")
    ax1.set_xticklabels([run.replace("_", " ").title() for run in runs], rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bar, count in zip(bars1, total_detections, strict=False):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(total_detections) * 0.01,
            f"{count}",
            ha="center",
            va="bottom",
        )

    # Detections per subject
    bars2 = ax2.bar(runs, detections_per_subject, color="purple", alpha=0.8)
    ax2.set_xlabel("Experimental Runs")
    ax2.set_ylabel("Detections per Subject")
    ax2.set_title("Average Detections per Subject")
    ax2.set_xticklabels([run.replace("_", " ").title() for run in runs], rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, avg in zip(bars2, detections_per_subject, strict=False):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(detections_per_subject) * 0.01,
            f"{avg:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plot_file = output_dir / "detection_stats_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved detection statistics plot: {plot_file}")


@beartype
def create_diagnosis_distribution_plot(
    all_metrics: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    """Create stacked bar plot showing diagnosis category distribution."""
    fig, ax = plt.subplots(figsize=(12, 8))

    runs = list(all_metrics.keys())
    categories = set()

    # Collect all categories
    for metrics in all_metrics.values():
        categories.update(
            metrics.get("diagnosis_metrics", {}).get("diagnosis_categories", {}).keys()
        )

    categories = sorted(categories)

    # Create data for stacked bar chart
    category_data = {cat: [] for cat in categories}

    for metrics in all_metrics.values():
        diag_cats = metrics.get("diagnosis_metrics", {}).get("diagnosis_categories", {})
        for cat in categories:
            category_data[cat].append(diag_cats.get(cat, 0))

    # Create stacked bar chart
    bottom = np.zeros(len(runs))
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

    for i, (cat, data) in enumerate(category_data.items()):
        ax.bar(
            runs,
            data,
            bottom=bottom,
            label=cat.replace("_", " ").title(),
            color=colors[i],
            alpha=0.8,
        )
        bottom += np.array(data)

    ax.set_xlabel("Experimental Runs")
    ax.set_ylabel("Number of Diagnoses")
    ax.set_title("Diagnosis Category Distribution Across Runs")
    ax.set_xticklabels([run.replace("_", " ").title() for run in runs], rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / "diagnosis_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved diagnosis distribution plot: {plot_file}")


@beartype
def compare_runs(parent_dir: Path, output_dir: Path, run_names: list[str] | None = None) -> None:
    """Compare metrics from multiple experimental runs."""
    logger.info(f"📊 Comparing runs from {parent_dir}")

    # Load all metrics
    all_metrics = load_metrics_from_runs(parent_dir, run_names)

    if not all_metrics:
        raise ValueError("No metrics found to compare")

    logger.info(f"Found metrics for {len(all_metrics)} runs: {list(all_metrics.keys())}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison table
    comparison_df = create_comparison_table(all_metrics)
    table_file = output_dir / "comparison_table.csv"
    comparison_df.to_csv(table_file, index=False)
    logger.info(f"Saved comparison table: {table_file}")

    # Create plots
    create_success_rate_plot(all_metrics, output_dir)
    create_detection_stats_plot(all_metrics, output_dir)
    create_diagnosis_distribution_plot(all_metrics, output_dir)

    # Save combined metrics
    combined_file = output_dir / "combined_metrics.json"
    with open(combined_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved combined metrics: {combined_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("\n" + "=" * 60)
    print(f"📁 All results saved to: {output_dir}")
    print("=" * 60)


@beartype
def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare multiple NOVA experimental runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--parent-dir",
        required=True,
        help="Parent directory containing multiple run metrics",
    )

    parser.add_argument(
        "--output",
        help="Output directory for comparison results",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names to compare (e.g., baseline agentic baseline_reasoning)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    try:
        # Generate output directory if not specified
        if args.output:
            output_dir = Path(args.output)
        else:
            parent_dir = Path(args.parent_dir)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_dir = parent_dir / f"comparison_{timestamp}"

        # Run comparison
        compare_runs(parent_dir=Path(args.parent_dir), output_dir=output_dir, run_names=args.models)

    except KeyboardInterrupt:
        logger.info("🛑 Comparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Comparison failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
