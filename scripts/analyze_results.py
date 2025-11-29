#!/usr/bin/env python3
"""NOVA Results Analysis and Visualization

Analyzes results from multiple NOVA evaluation runs and creates plots and tables
for research papers.

Usage:
    python scripts/analyze_results.py --input-dir ./runs --output-dir ./paper_results
    python scripts/analyze_results.py --input-dir ./runs --output-dir ./paper_results --compare-models
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
def load_results(input_dir: str | Path) -> dict[str, Any]:
    """Load all evaluation results from input directory.

    Args:
        input_dir: Directory containing evaluation results

    Returns:
        Dictionary with all results organized by experiment
    """
    input_dir = Path(input_dir)
    results = {}

    # Look for result files recursively
    for result_file in input_dir.rglob("*.json"):
        if "result.json" in result_file.name or "config.json" in result_file.name:
            continue  # Skip individual result/config files

        try:
            with open(result_file) as f:
                data = json.load(f)

            # Extract experiment name from path
            relative_path = result_file.relative_to(input_dir)
            experiment_name = str(relative_path.parent)

            results[experiment_name] = data
            logger.info(f"Loaded results from {experiment_name}")

        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")

    return results


@beartype
def extract_metrics(results: dict[str, Any]) -> pd.DataFrame:
    """Extract performance metrics into a DataFrame.

    Args:
        results: Dictionary of evaluation results

    Returns:
        DataFrame with metrics organized by experiment
    """
    metrics_data = []

    for experiment_name, result in results.items():
        if "metrics" not in result:
            continue

        metrics = result["metrics"]
        config = result.get("config", {})

        # Extract configuration details
        model = config.get("model", {}).get("name", "unknown")
        agentic = config.get("agentic", {}).get("enabled", False)
        use_tools = config.get("agentic", {}).get("use_tools", False)

        # Extract task-specific metrics
        row = {
            "experiment": experiment_name,
            "model": model,
            "agentic": agentic,
            "use_tools": use_tools,
            "localization_map": metrics.get("localization", {}).get("map50", 0.0),
            "diagnosis_accuracy": metrics.get("diagnosis", {}).get("accuracy", 0.0),
            "diagnosis_top5": metrics.get("diagnosis", {}).get("top5", 0.0),
            "caption_bleu": metrics.get("caption", {}).get("bleu", 0.0),
            "caption_bertscore": metrics.get("caption", {}).get("bert_score", 0.0),
            "avg_performance": 0.0,  # Will be calculated
        }

        # Calculate average performance (exclude empty metrics)
        valid_metrics = [
            row["localization_map"],
            row["diagnosis_accuracy"],
            row["diagnosis_top5"],
            row["caption_bleu"],
            row["caption_bertscore"]
        ]
        row["avg_performance"] = np.mean([m for m in valid_metrics if m > 0])

        metrics_data.append(row)

    df = pd.DataFrame(metrics_data)
    logger.info(f"Extracted metrics for {len(df)} experiments")

    return df


@beartype
def create_performance_plot(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Create performance comparison plots.

    Args:
        df: DataFrame with performance metrics
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model comparison plot
    plt.figure(figsize=(12, 8))

    # Group by model
    model_groups = df.groupby("model")

    x = np.arange(len(model_groups))
    width = 0.2

    # Create subplot for each metric type
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("NOVA Dataset Performance by Model and Configuration", fontsize=16)

    metrics = [
        ("localization_map", "Localization mAP@0.5"),
        ("diagnosis_accuracy", "Diagnosis Accuracy"),
        ("caption_bleu", "Caption BLEU"),
        ("caption_bertscore", "Caption BERTScore")
    ]

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Create grouped bar chart
        models = []
        baseline_scores = []
        agentic_scores = []

        for model_name, group in model_groups:
            models.append(model_name.split('/')[-1])  # Get short model name

            baseline = group[~group["agentic"]][metric].mean()
            agentic = group[group["agentic"]][metric].mean()

            baseline_scores.append(baseline if not np.isnan(baseline) else 0)
            agentic_scores.append(agentic if not np.isnan(agentic) else 0)

        x_pos = np.arange(len(models))
        width = 0.35

        ax.bar(x_pos - width/2, baseline_scores, width, label='Baseline', alpha=0.8)
        ax.bar(x_pos + width/2, agentic_scores, width, label='Agentic', alpha=0.8)

        ax.set_xlabel('Model')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "performance_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved performance plot to {plot_path}")


@beartype
def create_latex_table(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Create LaTeX tables for paper inclusion.

    Args:
        df: DataFrame with performance metrics
        output_dir: Output directory for tables
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main results table
    table_path = output_dir / "results_table.tex"

    with open(table_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{NOVA Dataset Performance Results}\n")
        f.write("\\label{tab:nova_results}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Config & Loc. mAP & Diag. Acc. & Cap. BLEU & Avg. \\\\\n")
        f.write("\\midrule\n")

        for _, row in df.iterrows():
            model_short = row["model"].split('/')[-1]
            config = "Agentic" if row["agentic"] else "Baseline"
            if row["use_tools"] and row["agentic"]:
                config += "+T"

            f.write(f"{model_short} & {config} & ")
            f.write(f"{row['localization_map']:.3f} & ")
            f.write(f"{row['diagnosis_accuracy']:.3f} & ")
            f.write(f"{row['caption_bleu']:.3f} & ")
            f.write(f"{row['avg_performance']:.3f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    logger.info(f"Saved LaTeX table to {table_path}")

    # Model comparison table
    model_comparison_path = output_dir / "model_comparison_table.tex"

    # Create summary by model
    model_summary = df.groupby("model").agg({
        "localization_map": "max",
        "diagnosis_accuracy": "max",
        "caption_bleu": "max",
        "avg_performance": "max"
    }).reset_index()

    with open(model_comparison_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Model Performance Comparison (Best Configuration)}\n")
        f.write("\\label{tab:model_comparison}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Loc. mAP & Diag. Acc. & Cap. BLEU & Avg. \\\\\n")
        f.write("\\midrule\n")

        for _, row in model_summary.iterrows():
            model_short = row["model"].split('/')[-1]
            f.write(f"{model_short} & ")
            f.write(f"{row['localization_map']:.3f} & ")
            f.write(f"{row['diagnosis_accuracy']:.3f} & ")
            f.write(f"{row['caption_bleu']:.3f} & ")
            f.write(f"{row['avg_performance']:.3f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    logger.info(f"Saved model comparison table to {model_comparison_path}")


@beartype
def generate_summary_stats(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate summary statistics and save to JSON.

    Args:
        df: DataFrame with performance metrics
        output_dir: Output directory for summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_experiments": len(df),
        "models_tested": df["model"].nunique(),
        "agentic_experiments": df["agentic"].sum(),
        "baseline_experiments": (~df["agentic"]).sum(),
        "best_performance": {
            "overall": df.loc[df["avg_performance"].idxmax()].to_dict(),
            "localization": df.loc[df["localization_map"].idxmax()].to_dict(),
            "diagnosis": df.loc[df["diagnosis_accuracy"].idxmax()].to_dict(),
            "caption": df.loc[df["caption_bleu"].idxmax()].to_dict(),
        },
        "performance_stats": {
            "localization_map": {
                "mean": df["localization_map"].mean(),
                "std": df["localization_map"].std(),
                "min": df["localization_map"].min(),
                "max": df["localization_map"].max(),
            },
            "diagnosis_accuracy": {
                "mean": df["diagnosis_accuracy"].mean(),
                "std": df["diagnosis_accuracy"].std(),
                "min": df["diagnosis_accuracy"].min(),
                "max": df["diagnosis_accuracy"].max(),
            },
            "caption_bleu": {
                "mean": df["caption_bleu"].mean(),
                "std": df["caption_bleu"].std(),
                "min": df["caption_bleu"].min(),
                "max": df["caption_bleu"].max(),
            }
        }
    }

    summary_path = output_dir / "summary_stats.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Saved summary statistics to {summary_path}")


@beartype
def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze NOVA evaluation results and create visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing evaluation results",
    )

    parser.add_argument(
        "--output-dir",
        default="./paper_results",
        help="Output directory for analysis results",
    )

    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Create model comparison analysis",
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

    logger.info("📊 NOVA Results Analysis")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")

    try:
        # Load results
        results = load_results(args.input_dir)

        if not results:
            logger.error("No results found in input directory")
            sys.exit(1)

        # Extract metrics
        df = extract_metrics(results)

        if df.empty:
            logger.error("No valid metrics found in results")
            sys.exit(1)

        # Create outputs
        create_performance_plot(df, args.output_dir)
        create_latex_table(df, args.output_dir)
        generate_summary_stats(df, args.output_dir)

        # Save metrics DataFrame
        df_path = Path(args.output_dir) / "metrics.csv"
        df.to_csv(df_path, index=False)
        logger.info(f"Saved metrics to {df_path}")

        logger.info("✅ Analysis completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"💥 Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()