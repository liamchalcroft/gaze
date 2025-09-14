#!/usr/bin/env python3
"""
Master script for analyzing benchmark results.

This script orchestrates the entire results analysis pipeline:
1. Gather and aggregate results from benchmark runs
2. Generate LaTeX tables
3. Create publication-ready figures
4. Generate a comprehensive report
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any
from typing import cast

import pandas as pd
from loguru import logger


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"Running: {description}")
    logger.debug(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.success(f"✓ {description} completed successfully")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False


def generate_summary_report(results_file: Path, output_dir: Path) -> None:
    """Generate a summary report with key findings."""

    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return

    df: pd.DataFrame = pd.read_csv(results_file)  # type: ignore[assignment]

    # Generate summary statistics with explicit type handling
    summary: dict[str, int | float | list[str]] = {
        "total_combinations": len(df),
        "approaches": sorted([str(x) for x in df["approach"].astype(str).unique()]),  # type: ignore[union-attr]
        "tasks": sorted([str(x) for x in df["task"].astype(str).unique()]),  # type: ignore[union-attr]
        "models": sorted([str(x) for x in df["model"].astype(str).unique()]),  # type: ignore[union-attr]
        "total_samples": int(df["sample_count"].astype(float).sum()),  # type: ignore[union-attr]
        "failed_samples": int(df["failed_samples"].astype(float).sum()),  # type: ignore[union-attr]
        "success_rate": float(
            (df["sample_count"].astype(float).sum() - df["failed_samples"].astype(float).sum())
            / df["sample_count"].astype(float).sum()
            * 100  # type: ignore[union-attr]
        ),
    }

    # Best performing approach for each task
    best_performers: dict[str, dict[str, Any]] = {}
    tasks_list = summary["tasks"]
    if isinstance(tasks_list, list):
        for task in tasks_list:
            # task is already a string from the summary tasks list
            task_data = df[df["task"] == task]  # type: ignore[assignment]  # type: ignore[assignment]

            if task == "localization":
                best_idx = task_data["iou"].astype(float).idxmax()  # type: ignore[union-attr]
                best_approach = str(task_data.loc[best_idx, "approach"])  # type: ignore[union-attr]
                best_score = float(task_data.loc[best_idx, "iou"]) * 100  # type: ignore[union-attr]
                metric_name = "IoU"
            elif task == "caption":
                best_idx = task_data["bleu"].astype(float).idxmax()  # type: ignore[union-attr]
                best_approach = str(task_data.loc[best_idx, "approach"])  # type: ignore[union-attr]
                best_score = float(task_data.loc[best_idx, "bleu"])  # type: ignore[union-attr]
                metric_name = "BLEU"
            elif task == "diagnosis":
                best_idx = task_data["accuracy"].astype(float).idxmax()  # type: ignore[union-attr]
                best_approach = str(task_data.loc[best_idx, "approach"])  # type: ignore[union-attr]
                best_score = float(task_data.loc[best_idx, "accuracy"]) * 100  # type: ignore[union-attr]
                metric_name = "Accuracy"
            else:
                continue

            best_performers[task] = {
                "approach": best_approach,
                "score": best_score,
                "metric": metric_name,
            }

    # Calculate approach rankings
    approach_scores: dict[str, float] = {}
    approaches_list = summary["approaches"]
    if isinstance(approaches_list, list):
        for approach in approaches_list:
            # approach is already a string from the summary approaches list
            approach_data = df[df["approach"] == approach]  # type: ignore[assignment]

            # Calculate average normalized scores across tasks
            scores: list[float] = []
            tasks_list = summary["tasks"]
            if isinstance(tasks_list, list):
                for task in tasks_list:
                    # task is already a string from the summary tasks list
                    task_data = approach_data[approach_data["task"] == task]  # type: ignore[index]
                    if not task_data.empty:  # type: ignore[union-attr]
                        if task == "localization":
                            scores.append(float(task_data["iou"].astype(float).mean()))  # type: ignore[union-attr]
                        elif task == "caption":
                            scores.append(
                                float(task_data["bleu"].astype(float).mean()) / 100  # type: ignore[union-attr]
                            )  # Normalize BLEU to [0,1]
                        elif task == "diagnosis":
                            scores.append(float(task_data["accuracy"].astype(float).mean()))  # type: ignore[union-attr]

            approach_scores[approach] = sum(scores) / len(scores) if scores else 0.0

    # Rank approaches
    ranked_approaches = sorted(
        [(str(k), float(v)) for k, v in approach_scores.items()], key=lambda x: x[1], reverse=True
    )

    # Generate Markdown report
    report_lines = [
        "# NOVA Benchmark Results Analysis Report",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        f"- **Total benchmark combinations:** {summary['total_combinations']}",
        f"- **Approaches evaluated:** {len(cast(list, summary['approaches']))} "
        f"({', '.join([str(x) for x in cast(list, summary['approaches']) if isinstance(x, str)])})",  # type: ignore[arg-type]
        f"- **Tasks:** {len(cast(list, summary['tasks']))} "
        f"({', '.join([str(x) for x in cast(list, summary['tasks']) if isinstance(x, str)])})",  # type: ignore[arg-type]
        f"- **Models:** {len(cast(list, summary['models']))} "
        f"({', '.join([str(x) for x in cast(list, summary['models']) if isinstance(x, str)])})",  # type: ignore[arg-type]
        f"- **Total samples processed:** {summary['total_samples']:,}",
        f"- **Overall success rate:** {summary['success_rate']:.1f}%",
        "",
        "## Best Performing Approaches by Task",
        "",
    ]

    for task, best in best_performers.items():
        report_lines.append(f"### {task.title()}")
        report_lines.append(f"- **Winner:** {best['approach']}")
        score_text = (
            f"- **Score:** {best['score']:.1f}"
            f"{'%' if best['metric'] != 'BLEU' else ''} ({best['metric']})"
        )
        report_lines.append(score_text)
        report_lines.append("")

    report_lines.extend(
        [
            "## Overall Approach Rankings",
            "",
            "Ranked by average normalized performance across all tasks:",
            "",
        ]
    )

    for i, (approach, score) in enumerate(ranked_approaches, 1):
        report_lines.append(f"{i}. **{str(approach).title()}** - {float(score):.3f}")

    report_lines.extend(["", "## Task-Specific Performance Summary", ""])

    # Add task-specific summaries
    for task in cast(list, summary["tasks"]):  # type: ignore[arg-type]
        if not isinstance(task, str):
            continue
        task_data = df[df["task"] == task]  # type: ignore[assignment]
        report_lines.append(f"### {task.title()}")
        report_lines.append("")

        if task == "localization":
            report_lines.append("| Approach | mAP@30 | mAP@50 | mAP@50:95 | IoU |")
            report_lines.append("|----------|--------|--------|-----------|-----|")
            for approach in cast(list, summary["approaches"]):  # type: ignore[arg-type]
                # approach is already a string from the summary approaches list
                approach_task_data = task_data[task_data["approach"] == approach]  # type: ignore[index]
                if not approach_task_data.empty:  # type: ignore[union-attr]
                    map30 = (
                        approach_task_data["map30"].astype(float).mean() * 100  # type: ignore[union-attr]
                        if "map30" in approach_task_data.columns
                        else 0
                    )
                    map50 = (
                        approach_task_data["map50"].astype(float).mean() * 100  # type: ignore[union-attr]
                        if "map50" in approach_task_data.columns
                        else 0
                    )
                    map50_95 = (
                        approach_task_data["map50_95"].astype(float).mean() * 100  # type: ignore[union-attr]
                        if "map50_95" in approach_task_data.columns
                        else 0
                    )
                    iou = (
                        approach_task_data["iou"].astype(float).mean() * 100  # type: ignore[union-attr]
                        if "iou" in approach_task_data.columns
                        else 0
                    )
                    row_text = (
                        f"| {approach} | {map30:.1f}% | {map50:.1f}% | "
                        f"{map50_95:.1f}% | {iou:.1f}% |"
                    )
                    report_lines.append(row_text)

        elif task == "caption":
            report_lines.append("| Approach | BLEU | METEOR | BERT-F1 | RadGraph-F1 |")
            report_lines.append("|----------|------|--------|---------|-------------|")
            for approach in cast(list, summary["approaches"]):  # type: ignore[arg-type]
                # approach is already a string from the summary approaches list
                approach_task_data = task_data[task_data["approach"] == approach]  # type: ignore[index]
                if not approach_task_data.empty:  # type: ignore[union-attr]
                    bleu = (
                        approach_task_data["bleu"].astype(float).mean()  # type: ignore[union-attr]
                        if "bleu" in approach_task_data.columns
                        else 0
                    )
                    meteor = (
                        approach_task_data["meteor"].astype(float).mean()  # type: ignore[union-attr]
                        if "meteor" in approach_task_data.columns
                        else 0
                    )
                    bert_f1 = (
                        approach_task_data["bert_f1"].astype(float).mean()  # type: ignore[union-attr]
                        if "bert_f1" in approach_task_data.columns
                        else 0
                    )
                    radgraph_f1 = (
                        approach_task_data["radgraph_f1"].astype(float).mean()  # type: ignore[union-attr]
                        if "radgraph_f1" in approach_task_data.columns
                        else 0
                    )
                    row_text = (
                        f"| {approach} | {bleu:.3f} | {meteor:.3f} | "
                        f"{bert_f1:.3f} | {radgraph_f1:.3f} |"
                    )
                    report_lines.append(row_text)

        elif task == "diagnosis":
            report_lines.append("| Approach | Top-1 | Top-5 | Coverage | Entropy |")
            report_lines.append("|----------|-------|-------|----------|---------|")
            for approach in cast(list, summary["approaches"]):  # type: ignore[arg-type]
                # approach is already a string from the summary approaches list
                approach_task_data = task_data[task_data["approach"] == approach]  # type: ignore[index]
                if not approach_task_data.empty:  # type: ignore[union-attr]
                    top1 = (
                        approach_task_data["top1"].astype(float).mean() * 100  # type: ignore[union-attr]
                        if "top1" in approach_task_data.columns
                        else 0
                    )
                    top5 = (
                        approach_task_data["top5"].astype(float).mean() * 100  # type: ignore[union-attr]
                        if "top5" in approach_task_data.columns
                        else 0
                    )
                    coverage = (
                        approach_task_data["coverage"].astype(float).mean()  # type: ignore[union-attr]
                        if "coverage" in approach_task_data.columns
                        else 0
                    )
                    entropy = (
                        approach_task_data["entropy"].astype(float).mean()  # type: ignore[union-attr]
                        if "entropy" in approach_task_data.columns
                        else 0
                    )
                    row_text = (
                        f"| {approach} | {top1:.1f}% | {top5:.1f}% | "
                        f"{coverage:.3f} | {entropy:.3f} |"
                    )
                    report_lines.append(row_text)

        report_lines.append("")

    report_lines.extend(
        [
            "## Files Generated",
            "",
            "### LaTeX Tables",
            "- `outputs/tables/performance_table.tex` - Main performance comparison table",
            "- `outputs/tables/detailed_table.tex` - Detailed metrics table",
            "- `outputs/tables/task_breakdown_table.tex` - Task-specific breakdown",
            "- `outputs/tables/summary_table.tex` - Summary statistics",
            "",
            "### Figures",
            "- `outputs/figures/performance_comparison.png` - Bar chart comparison",
            "- `outputs/figures/radar_plot.png` - Multi-dimensional radar plot",
            "- `outputs/figures/performance_heatmap.png` - Performance heatmap",
            "- `outputs/figures/sample_counts.png` - Sample count and success rates",
            "- `outputs/figures/localization_comparison.png` - Localization-specific metrics",
            "- `outputs/figures/caption_comparison.png` - Caption-specific metrics",
            "- `outputs/figures/diagnosis_comparison.png` - Diagnosis-specific metrics",
            "",
            "### Data Files",
            "- `outputs/results_analysis/aggregated_results.csv` - Raw aggregated data",
            "- `outputs/results_analysis/aggregated_results.json` - JSON format results",
            "- `outputs/results_analysis/summary_by_approach.csv` - Summary statistics by approach",
            "",
            "---",
            "",
            "*Report generated by the NOVA Benchmark Results Analysis Pipeline*",
        ]
    )

    # Write report
    report_file = output_dir / "benchmark_analysis_report.md"
    with report_file.open("w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Summary report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and generate tables/figures"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="runs/full_benchmark",
        help="Directory containing benchmark results",
    )
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument(
        "--skip-gather",
        action="store_true",
        help="Skip gathering results (use existing aggregated data)",
    )
    parser.add_argument("--skip-tables", action="store_true", help="Skip LaTeX table generation")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    parser.add_argument(
        "--figure-format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for figures",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Configure logging
    if not args.verbose:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return 1

    logger.info("🔍 Starting benchmark results analysis")
    logger.info(f"📂 Results directory: {results_dir}")
    logger.info(f"📁 Output directory: {output_dir}")

    success_count = 0
    total_steps = 5  # gather, tables, statistical_analysis, figures, report

    # Step 1: Gather and aggregate results
    if not args.skip_gather:
        logger.info("📊 Step 1/4: Gathering and aggregating results...")
        cmd = [
            sys.executable,
            "scripts/gather_results.py",
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(output_dir / "results_analysis"),
        ]
        if args.verbose:
            cmd.append("--verbose")

        if run_command(cmd, "Gathering results"):
            success_count += 1
    else:
        logger.info("⏭️  Step 1/4: Skipping results gathering (using existing data)")
        success_count += 1

    # Check if aggregated results exist
    aggregated_results_file = output_dir / "results_analysis" / "aggregated_results.csv"
    if not aggregated_results_file.exists():
        logger.error(f"Aggregated results file not found: {aggregated_results_file}")
        logger.error("Cannot proceed without aggregated results. Run without --skip-gather.")
        return 1

    # Step 2: Generate LaTeX tables
    if not args.skip_tables:
        logger.info("📝 Step 2/5: Generating LaTeX tables...")
        cmd = [
            sys.executable,
            "scripts/generate_latex_tables.py",
            "--results-file",
            str(aggregated_results_file),
            "--output-dir",
            str(output_dir / "tables"),
            "--stats-dir",
            str(output_dir / "statistical_analysis"),
        ]

        if run_command(cmd, "Generating LaTeX tables"):
            success_count += 1
    else:
        logger.info("⏭️  Step 2/5: Skipping LaTeX table generation")
        success_count += 1

    # Step 3: Perform statistical analysis
    logger.info("📊 Step 3/5: Performing statistical significance testing...")
    cmd = [
        sys.executable,
        "scripts/statistical_analysis.py",
        "--results-dir",
        str(results_dir),
        "--output-dir",
        str(output_dir / "statistical_analysis"),
        "--correction-method",
        "fdr",
    ]
    if args.verbose:
        cmd.append("--verbose")

    if run_command(cmd, "Statistical significance testing"):
        success_count += 1

    # Step 4: Generate figures
    if not args.skip_figures:
        logger.info("📈 Step 4/5: Generating figures...")
        cmd = [
            sys.executable,
            "scripts/generate_figures.py",
            "--results-file",
            str(aggregated_results_file),
            "--output-dir",
            str(output_dir / "figures"),
            "--format",
            args.figure_format,
        ]

        if run_command(cmd, "Generating figures"):
            success_count += 1
    else:
        logger.info("⏭️  Step 4/5: Skipping figure generation")
        success_count += 1

    # Step 5: Generate summary report
    logger.info("📋 Step 5/5: Generating summary report...")
    try:
        generate_summary_report(aggregated_results_file, output_dir)
        success_count += 1
    except Exception as e:
        logger.error(f"Failed to generate summary report: {e}")

    # Final summary
    logger.info(f"\n🎉 Analysis complete! {success_count}/{total_steps} steps successful")

    if success_count == total_steps:
        logger.success("✅ All analysis steps completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info("🔗 Key outputs:")
        logger.info(f"  - Summary report: {output_dir}/benchmark_analysis_report.md")
        logger.info(f"  - LaTeX tables: {output_dir}/tables/")
        logger.info(f"  - Figures: {output_dir}/figures/")
        logger.info(f"  - Raw data: {output_dir}/results_analysis/")
        return 0
    else:
        logger.warning(f"⚠️  Some steps failed ({success_count}/{total_steps} successful)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
