#!/usr/bin/env python3
"""
Generate LaTeX tables from aggregated benchmark results.

This script reads the aggregated results and generates publication-ready LaTeX tables.
"""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

# Mapping from internal approach names to display names for the table
APPROACH_DISPLAY_NAMES = {
    "baseline": "Baseline",
    "comprehensive": "Reasoning + Web + Visual",
    "multiturn": "Reasoning",
    "visual": "Reasoning + Visual",
    "web_search": "Reasoning + Web",
    # Add retrieval when it appears
    "retrieval": "Retrieval-Augmented Agent",
}

# Ordering for approaches in the table
APPROACH_ORDER = ["baseline", "multiturn", "web_search", "visual", "comprehensive"]


def load_aggregated_results(results_file: Path) -> pd.DataFrame:
    """Load the aggregated results CSV file."""
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    return pd.read_csv(results_file)


def load_statistical_results(stats_dir: Path) -> pd.DataFrame | None:
    """Load statistical significance test results."""
    stats_file = stats_dir / "significant_results.csv"
    if not stats_file.exists():
        logger.warning(f"Statistical results file not found: {stats_file}")
        return None

    return pd.read_csv(stats_file)


def is_significantly_better(
    approach_a: str, approach_b: str, task: str, metric: str, stats_df: pd.DataFrame | None
) -> bool:
    """Check if approach_a is significantly better than approach_b for a given metric."""
    if stats_df is None:
        return False

    # Map internal metric names to statistical test metric names
    metric_mapping = {
        "map50": "detection_mAP50",
        "map30": "detection_mAP30",
        "map50_95": "detection_mAP50_95",
        "bleu": "caption_bleu",
        "meteor": "caption_meteor",
        "bert_f1": "caption_bert_f1",
        "radgraph_f1": "caption_radgraph_f1",
        "top1": "diagnosis_top1",
        "top5": "diagnosis_top5",
    }

    stat_metric = metric_mapping.get(metric, metric)

    # Check if there's a significant result for this comparison
    result = stats_df[
        (stats_df["approach_a"] == approach_a)
        & (stats_df["approach_b"] == approach_b)
        & (stats_df["task"] == task)
        & (stats_df["metric"] == stat_metric)
        & (stats_df["significant"])
    ]

    # Also check the reverse comparison
    if result.empty:
        result = stats_df[
            (stats_df["approach_a"] == approach_b)
            & (stats_df["approach_b"] == approach_a)
            & (stats_df["task"] == task)
            & (stats_df["metric"] == stat_metric)
            & (stats_df["significant"])
        ]

    return not result.empty


def format_metric_value(
    value: float, metric_type: str = "percentage", is_significant: bool = False
) -> str:
    """Format a metric value for display in LaTeX table."""
    if pd.isna(value) or value == 0.0:
        return "--"

    if metric_type == "percentage":
        # Convert to percentage and format with 1 decimal place
        formatted = f"{value * 100:.1f}"
    elif metric_type == "score":
        # Format as score with 3 decimal places
        formatted = f"{value:.3f}"
    else:
        # Default formatting
        formatted = f"{value:.2f}"

    # Add significance marker
    if is_significant:
        formatted += "*"

    return formatted


def generate_performance_table(
    df: pd.DataFrame, output_file: Path, stats_df: pd.DataFrame | None = None
) -> None:
    """Generate the main performance comparison table."""

    # Create summary by approach (average across models)
    summary_data = {}

    for approach in APPROACH_ORDER:
        if approach not in df["approach"].values:
            logger.warning(f"Approach '{approach}' not found in results")
            continue

        approach_data = df[df["approach"] == approach]

        # Get metrics for each task (handle missing tasks)
        loc_data = approach_data[approach_data["task"] == "localization"]
        cap_data = approach_data[approach_data["task"] == "caption"]
        diag_data = approach_data[approach_data["task"] == "diagnosis"]

        localization_map50 = (
            loc_data["map50"].mean()
            if not loc_data.empty and "map50" in loc_data.columns
            else float("nan")
        )
        caption_bleu = (
            cap_data["bleu"].mean()
            if not cap_data.empty and "bleu" in cap_data.columns
            else float("nan")
        )
        diagnosis_top1 = (
            diag_data["top1"].mean()
            if not diag_data.empty and "top1" in diag_data.columns
            else float("nan")
        )

        summary_data[approach] = {
            "localization_map50": localization_map50,
            "caption_bleu": caption_bleu,
            "diagnosis_top1": diagnosis_top1,
        }

    # Generate LaTeX table
    latex_lines = [
        "\\begin{table}[t]",
        "  \\caption{Performance comparison on NOVA benchmark (higher is better).}",
        "  \\label{tab:performance}",
        "  \\centering",
        "  \\begin{tabular}{lccc}",
        "    \\hline",
        "    \\textbf{Agent Configuration} & \\textbf{Localisation mAP50} & \\textbf{Caption BLEU} & \\textbf{Diagnosis Top-1}\\\\",
        "    \\hline",
    ]

    # Find best values for bolding
    best_map50 = max(
        summary_data[a]["localization_map50"]
        for a in summary_data
        if not pd.isna(summary_data[a]["localization_map50"])
    )
    best_bleu = max(
        summary_data[a]["caption_bleu"]
        for a in summary_data
        if not pd.isna(summary_data[a]["caption_bleu"])
    )
    best_top1 = max(
        summary_data[a]["diagnosis_top1"]
        for a in summary_data
        if not pd.isna(summary_data[a]["diagnosis_top1"])
    )

    # Add rows for each approach
    for approach in APPROACH_ORDER:
        if approach not in summary_data:
            continue

        display_name = APPROACH_DISPLAY_NAMES.get(approach, approach.title())
        data = summary_data[approach]

        # Check for significance vs baseline and if it's the best score
        is_map50_significant = stats_df is not None and is_significantly_better(
            approach, "baseline", "localization", "map50", stats_df
        )
        is_bleu_significant = stats_df is not None and is_significantly_better(
            approach, "baseline", "caption", "bleu", stats_df
        )
        is_top1_significant = stats_df is not None and is_significantly_better(
            approach, "baseline", "diagnosis", "top1", stats_df
        )

        map50_str = format_metric_value(
            data["localization_map50"], "percentage", is_map50_significant
        )
        bleu_str = format_metric_value(data["caption_bleu"], "score", is_bleu_significant)
        top1_str = format_metric_value(data["diagnosis_top1"], "percentage", is_top1_significant)

        # Bold the best scores
        if (
            not pd.isna(data["localization_map50"])
            and abs(data["localization_map50"] - best_map50) < 1e-6
        ):
            map50_str = f"\\textbf{{{map50_str}}}"
        if not pd.isna(data["caption_bleu"]) and abs(data["caption_bleu"] - best_bleu) < 1e-6:
            bleu_str = f"\\textbf{{{bleu_str}}}"
        if not pd.isna(data["diagnosis_top1"]) and abs(data["diagnosis_top1"] - best_top1) < 1e-6:
            top1_str = f"\\textbf{{{top1_str}}}"

        latex_lines.append(f"    {display_name:<25} & {map50_str} & {bleu_str} & {top1_str}\\\\")

    # Add footnote for significance markers
    latex_lines.extend(
        [
            "    \\hline",
            "  \\end{tabular}",
            "" if stats_df is None else "  \\begin{flushleft}",
            ""
            if stats_df is None
            else "  \\footnotesize{* Statistically significant difference vs. baseline (p < 0.05, FDR corrected)}",
            "" if stats_df is None else "  \\end{flushleft}",
            "\\end{table}",
        ]
    )

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))

    logger.info(f"Performance table saved to {output_file}")


def generate_detailed_table(df: pd.DataFrame, output_file: Path) -> None:
    """Generate a detailed table with all metrics."""

    latex_lines = [
        "\\begin{table}[t]",
        "  \\caption{Detailed performance metrics on NOVA benchmark.}",
        "  \\label{tab:detailed_performance}",
        "  \\centering",
        "  \\begin{tabular}{lcccccc}",
        "    \\hline",
        "    \\textbf{Approach} & \\textbf{Loc. mAP50} & \\textbf{Cap. BLEU} & \\textbf{Cap. METEOR} & \\textbf{Diag. Top1} & \\textbf{Diag. Top5} & \\textbf{Samples}\\\\",
        "    \\hline",
    ]

    # Create summary by approach
    for approach in APPROACH_ORDER:
        if approach not in df["approach"].values:
            continue

        approach_data = df[df["approach"] == approach]
        display_name = APPROACH_DISPLAY_NAMES.get(approach, approach.title())

        # Calculate metrics
        loc_data = approach_data[approach_data["task"] == "localization"]
        cap_data = approach_data[approach_data["task"] == "caption"]
        diag_data = approach_data[approach_data["task"] == "diagnosis"]

        loc_map50 = (
            loc_data["map50"].mean()
            if not loc_data.empty and "map50" in loc_data.columns
            else float("nan")
        )
        cap_bleu = (
            cap_data["bleu"].mean()
            if not cap_data.empty and "bleu" in cap_data.columns
            else float("nan")
        )
        cap_meteor = (
            cap_data["meteor"].mean()
            if not cap_data.empty and "meteor" in cap_data.columns
            else float("nan")
        )
        diag_top1 = (
            diag_data["top1"].mean()
            if not diag_data.empty and "top1" in diag_data.columns
            else float("nan")
        )
        diag_top5 = (
            diag_data["top5"].mean()
            if not diag_data.empty and "top5" in diag_data.columns
            else float("nan")
        )
        total_samples = approach_data["sample_count"].sum()

        # Format values
        loc_str = format_metric_value(loc_map50, "percentage")
        bleu_str = format_metric_value(cap_bleu, "score")
        meteor_str = format_metric_value(cap_meteor, "score")
        top1_str = format_metric_value(diag_top1, "percentage")
        top5_str = format_metric_value(diag_top5, "percentage")

        latex_lines.append(
            f"    {display_name:<20} & {loc_str} & {bleu_str} & {meteor_str} & {top1_str} & {top5_str} & {total_samples}\\\\"
        )

    latex_lines.extend(["    \\hline", "  \\end{tabular}", "\\end{table}"])

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))

    logger.info(f"Detailed table saved to {output_file}")


def generate_task_breakdown_table(df: pd.DataFrame, output_file: Path) -> None:
    """Generate a table showing task-specific breakdowns."""

    latex_lines = [
        "\\begin{table*}[t]",
        "  \\caption{Task-specific performance breakdown on NOVA benchmark.}",
        "  \\label{tab:task_breakdown}",
        "  \\centering",
        "  \\begin{tabular}{l|ccc|cccc|ccc}",
        "    \\hline",
        "    & \\multicolumn{3}{c|}{\\textbf{Localization}} & \\multicolumn{4}{c|}{\\textbf{Caption}} & \\multicolumn{3}{c}{\\textbf{Diagnosis}} \\\\",
        "    \\textbf{Approach} & mAP30 & mAP50 & mAP50:95 & BLEU & METEOR & BERT-F1 & RadGraph & Top1 & Top5 & Coverage \\\\",
        "    \\hline",
    ]

    for approach in APPROACH_ORDER:
        if approach not in df["approach"].values:
            continue

        approach_data = df[df["approach"] == approach]
        display_name = APPROACH_DISPLAY_NAMES.get(approach, approach.title())

        # Get task-specific data
        loc_data = approach_data[approach_data["task"] == "localization"]
        cap_data = approach_data[approach_data["task"] == "caption"]
        diag_data = approach_data[approach_data["task"] == "diagnosis"]

        # Localization metrics
        map30 = format_metric_value(
            loc_data["map30"].mean()
            if not loc_data.empty and "map30" in loc_data.columns
            else float("nan"),
            "percentage",
        )
        map50 = format_metric_value(
            loc_data["map50"].mean()
            if not loc_data.empty and "map50" in loc_data.columns
            else float("nan"),
            "percentage",
        )
        map50_95 = format_metric_value(
            loc_data["map50_95"].mean()
            if not loc_data.empty and "map50_95" in loc_data.columns
            else float("nan"),
            "percentage",
        )

        # Caption metrics
        bleu = format_metric_value(
            cap_data["bleu"].mean()
            if not cap_data.empty and "bleu" in cap_data.columns
            else float("nan"),
            "score",
        )
        meteor = format_metric_value(
            cap_data["meteor"].mean()
            if not cap_data.empty and "meteor" in cap_data.columns
            else float("nan"),
            "score",
        )
        bert_f1 = format_metric_value(
            cap_data["bert_f1"].mean()
            if not cap_data.empty and "bert_f1" in cap_data.columns
            else float("nan"),
            "score",
        )
        radgraph = format_metric_value(
            cap_data["radgraph_f1"].mean()
            if not cap_data.empty and "radgraph_f1" in cap_data.columns
            else float("nan"),
            "score",
        )

        # Diagnosis metrics
        top1 = format_metric_value(
            diag_data["top1"].mean()
            if not diag_data.empty and "top1" in diag_data.columns
            else float("nan"),
            "percentage",
        )
        top5 = format_metric_value(
            diag_data["top5"].mean()
            if not diag_data.empty and "top5" in diag_data.columns
            else float("nan"),
            "percentage",
        )
        coverage = format_metric_value(
            diag_data["coverage"].mean()
            if not diag_data.empty and "coverage" in diag_data.columns
            else float("nan"),
            "score",
        )

        latex_lines.append(
            f"    {display_name:<20} & {map30} & {map50} & {map50_95} & {bleu} & {meteor} & {bert_f1} & {radgraph} & {top1} & {top5} & {coverage} \\\\"
        )

    latex_lines.extend(["    \\hline", "  \\end{tabular}", "\\end{table*}"])

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))

    logger.info(f"Task breakdown table saved to {output_file}")


def generate_summary_statistics(df: pd.DataFrame, output_file: Path) -> None:
    """Generate a summary statistics table."""

    latex_lines = [
        "\\begin{table}[t]",
        "  \\caption{Benchmark execution summary.}",
        "  \\label{tab:summary}",
        "  \\centering",
        "  \\begin{tabular}{lccc}",
        "    \\hline",
        "    \\textbf{Approach} & \\textbf{Total Samples} & \\textbf{Failed Samples} & \\textbf{Success Rate} \\\\",
        "    \\hline",
    ]

    for approach in APPROACH_ORDER:
        if approach not in df["approach"].values:
            continue

        approach_data = df[df["approach"] == approach]
        display_name = APPROACH_DISPLAY_NAMES.get(approach, approach.title())

        total_samples = approach_data["sample_count"].sum()
        failed_samples = approach_data["failed_samples"].sum()
        success_rate = (
            (total_samples - failed_samples) / total_samples if total_samples > 0 else 0.0
        )

        success_rate_str = format_metric_value(success_rate, "percentage")

        latex_lines.append(
            f"    {display_name:<25} & {total_samples} & {failed_samples} & {success_rate_str}\\% \\\\"
        )

    latex_lines.extend(["    \\hline", "  \\end{tabular}", "\\end{table}"])

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))

    logger.info(f"Summary statistics table saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from benchmark results")
    parser.add_argument(
        "--results-file",
        type=str,
        default="outputs/results_analysis/aggregated_results.csv",
        help="Path to aggregated results CSV file",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/tables", help="Directory to save LaTeX tables"
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default="outputs/statistical_analysis",
        help="Directory containing statistical test results",
    )
    parser.add_argument(
        "--table-type",
        choices=["all", "performance", "detailed", "breakdown", "summary"],
        default="all",
        help="Type of table to generate",
    )

    args = parser.parse_args()

    results_file = Path(args.results_file)
    output_dir = Path(args.output_dir)
    stats_dir = Path(args.stats_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    logger.info(f"Loading results from {results_file}")
    df = load_aggregated_results(results_file)

    # Load statistical results
    logger.info(f"Loading statistical results from {stats_dir}")
    stats_df = load_statistical_results(stats_dir)
    if stats_df is not None:
        logger.info(f"Loaded {len(stats_df)} significant statistical results")
    else:
        logger.warning(
            "No statistical results found - tables will not include significance markers"
        )

    logger.info(f"Loaded {len(df)} result rows")
    logger.info(f"Approaches: {sorted(df['approach'].unique())}")
    logger.info(f"Tasks: {sorted(df['task'].unique())}")
    logger.info(f"Models: {sorted(df['model'].unique())}")

    # Generate tables based on requested type
    if args.table_type in ["all", "performance"]:
        generate_performance_table(df, output_dir / "performance_table.tex", stats_df)

    if args.table_type in ["all", "detailed"]:
        generate_detailed_table(df, output_dir / "detailed_table.tex")

    if args.table_type in ["all", "breakdown"]:
        generate_task_breakdown_table(df, output_dir / "task_breakdown_table.tex")

    if args.table_type in ["all", "summary"]:
        generate_summary_statistics(df, output_dir / "summary_table.tex")

    logger.info(f"LaTeX tables generated in {output_dir}")


if __name__ == "__main__":
    main()
