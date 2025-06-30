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
from loguru import logger
import json
import pandas as pd

def run_command(command: list, description: str) -> bool:
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
    
    df = pd.read_csv(results_file)
    
    # Generate summary statistics
    summary = {
        "total_combinations": len(df),
        "approaches": sorted(df['approach'].unique().tolist()),
        "tasks": sorted(df['task'].unique().tolist()),
        "models": sorted(df['model'].unique().tolist()),
        "total_samples": df['sample_count'].sum(),
        "failed_samples": df['failed_samples'].sum(),
        "success_rate": (df['sample_count'].sum() - df['failed_samples'].sum()) / df['sample_count'].sum() * 100
    }
    
    # Best performing approach for each task
    best_performers = {}
    for task in summary["tasks"]:
        task_data = df[df['task'] == task]
        
        if task == 'localization':
            best_idx = task_data['iou'].idxmax()
            best_approach = task_data.loc[best_idx, 'approach']
            best_score = task_data.loc[best_idx, 'iou'] * 100
            metric_name = "IoU"
        elif task == 'caption':
            best_idx = task_data['bleu'].idxmax()
            best_approach = task_data.loc[best_idx, 'approach']
            best_score = task_data.loc[best_idx, 'bleu']
            metric_name = "BLEU"
        elif task == 'diagnosis':
            best_idx = task_data['accuracy'].idxmax()
            best_approach = task_data.loc[best_idx, 'approach']
            best_score = task_data.loc[best_idx, 'accuracy'] * 100
            metric_name = "Accuracy"
        else:
            continue
        
        best_performers[task] = {
            "approach": best_approach,
            "score": best_score,
            "metric": metric_name
        }
    
    # Calculate approach rankings
    approach_scores = {}
    for approach in summary["approaches"]:
        approach_data = df[df['approach'] == approach]
        
        # Calculate average normalized scores across tasks
        scores = []
        for task in summary["tasks"]:
            task_data = approach_data[approach_data['task'] == task]
            if not task_data.empty:
                if task == 'localization':
                    scores.append(task_data['iou'].mean())
                elif task == 'caption':
                    scores.append(task_data['bleu'].mean() / 100)  # Normalize BLEU to [0,1]
                elif task == 'diagnosis':
                    scores.append(task_data['accuracy'].mean())
        
        approach_scores[approach] = sum(scores) / len(scores) if scores else 0
    
    # Rank approaches
    ranked_approaches = sorted(approach_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Generate Markdown report
    report_lines = [
        "# NOVA Benchmark Results Analysis Report",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        f"- **Total benchmark combinations:** {summary['total_combinations']}",
        f"- **Approaches evaluated:** {len(summary['approaches'])} ({', '.join(summary['approaches'])})",
        f"- **Tasks:** {len(summary['tasks'])} ({', '.join(summary['tasks'])})",
        f"- **Models:** {len(summary['models'])} ({', '.join(summary['models'])})",
        f"- **Total samples processed:** {summary['total_samples']:,}",
        f"- **Overall success rate:** {summary['success_rate']:.1f}%",
        "",
        "## Best Performing Approaches by Task",
        ""
    ]
    
    for task, best in best_performers.items():
        report_lines.append(f"### {task.title()}")
        report_lines.append(f"- **Winner:** {best['approach']}")
        report_lines.append(f"- **Score:** {best['score']:.1f}{'%' if best['metric'] != 'BLEU' else ''} ({best['metric']})")
        report_lines.append("")
    
    report_lines.extend([
        "## Overall Approach Rankings",
        "",
        "Ranked by average normalized performance across all tasks:",
        ""
    ])
    
    for i, (approach, score) in enumerate(ranked_approaches, 1):
        report_lines.append(f"{i}. **{approach.title()}** - {score:.3f}")
    
    report_lines.extend([
        "",
        "## Task-Specific Performance Summary",
        ""
    ])
    
    # Add task-specific summaries
    for task in summary["tasks"]:
        task_data = df[df['task'] == task]
        report_lines.append(f"### {task.title()}")
        report_lines.append("")
        
        if task == 'localization':
            report_lines.append("| Approach | mAP@30 | mAP@50 | mAP@50:95 | IoU |")
            report_lines.append("|----------|--------|--------|-----------|-----|")
            for approach in summary["approaches"]:
                approach_task_data = task_data[task_data['approach'] == approach]
                if not approach_task_data.empty:
                    map30 = approach_task_data['map30'].mean() * 100 if 'map30' in approach_task_data.columns else 0
                    map50 = approach_task_data['map50'].mean() * 100 if 'map50' in approach_task_data.columns else 0
                    map50_95 = approach_task_data['map50_95'].mean() * 100 if 'map50_95' in approach_task_data.columns else 0
                    iou = approach_task_data['iou'].mean() * 100 if 'iou' in approach_task_data.columns else 0
                    report_lines.append(f"| {approach} | {map30:.1f}% | {map50:.1f}% | {map50_95:.1f}% | {iou:.1f}% |")
        
        elif task == 'caption':
            report_lines.append("| Approach | BLEU | METEOR | BERT-F1 | RadGraph-F1 |")
            report_lines.append("|----------|------|--------|---------|-------------|")
            for approach in summary["approaches"]:
                approach_task_data = task_data[task_data['approach'] == approach]
                if not approach_task_data.empty:
                    bleu = approach_task_data['bleu'].mean() if 'bleu' in approach_task_data.columns else 0
                    meteor = approach_task_data['meteor'].mean() if 'meteor' in approach_task_data.columns else 0
                    bert_f1 = approach_task_data['bert_f1'].mean() if 'bert_f1' in approach_task_data.columns else 0
                    radgraph_f1 = approach_task_data['radgraph_f1'].mean() if 'radgraph_f1' in approach_task_data.columns else 0
                    report_lines.append(f"| {approach} | {bleu:.3f} | {meteor:.3f} | {bert_f1:.3f} | {radgraph_f1:.3f} |")
        
        elif task == 'diagnosis':
            report_lines.append("| Approach | Top-1 | Top-5 | Coverage | Entropy |")
            report_lines.append("|----------|-------|-------|----------|---------|")
            for approach in summary["approaches"]:
                approach_task_data = task_data[task_data['approach'] == approach]
                if not approach_task_data.empty:
                    top1 = approach_task_data['top1'].mean() * 100 if 'top1' in approach_task_data.columns else 0
                    top5 = approach_task_data['top5'].mean() * 100 if 'top5' in approach_task_data.columns else 0
                    coverage = approach_task_data['coverage'].mean() if 'coverage' in approach_task_data.columns else 0
                    entropy = approach_task_data['entropy'].mean() if 'entropy' in approach_task_data.columns else 0
                    report_lines.append(f"| {approach} | {top1:.1f}% | {top5:.1f}% | {coverage:.3f} | {entropy:.3f} |")
        
        report_lines.append("")
    
    report_lines.extend([
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
        "*Report generated by the NOVA Benchmark Results Analysis Pipeline*"
    ])
    
    # Write report
    report_file = output_dir / "benchmark_analysis_report.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Summary report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results and generate tables/figures")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="runs/full_benchmark",
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory"
    )
    parser.add_argument(
        "--skip-gather",
        action="store_true",
        help="Skip gathering results (use existing aggregated data)"
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip LaTeX table generation"
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation"
    )
    parser.add_argument(
        "--figure-format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for figures"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
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
    
    logger.info(f"🔍 Starting benchmark results analysis")
    logger.info(f"📂 Results directory: {results_dir}")
    logger.info(f"📁 Output directory: {output_dir}")
    
    success_count = 0
    total_steps = 5  # gather, tables, statistical_analysis, figures, report
    
    # Step 1: Gather and aggregate results
    if not args.skip_gather:
        logger.info("📊 Step 1/4: Gathering and aggregating results...")
        cmd = [
            sys.executable, "scripts/gather_results.py",
            "--results-dir", str(results_dir),
            "--output-dir", str(output_dir / "results_analysis")
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
            sys.executable, "scripts/generate_latex_tables.py",
            "--results-file", str(aggregated_results_file),
            "--output-dir", str(output_dir / "tables"),
            "--stats-dir", str(output_dir / "statistical_analysis")
        ]
        
        if run_command(cmd, "Generating LaTeX tables"):
            success_count += 1
    else:
        logger.info("⏭️  Step 2/5: Skipping LaTeX table generation")
        success_count += 1
    
    # Step 3: Perform statistical analysis
    logger.info("📊 Step 3/5: Performing statistical significance testing...")
    cmd = [
        sys.executable, "scripts/statistical_analysis.py",
        "--results-dir", str(results_dir),
        "--output-dir", str(output_dir / "statistical_analysis"),
        "--correction-method", "fdr"
    ]
    if args.verbose:
        cmd.append("--verbose")
    
    if run_command(cmd, "Statistical significance testing"):
        success_count += 1
    
    # Step 4: Generate figures
    if not args.skip_figures:
        logger.info("📈 Step 4/5: Generating figures...")
        cmd = [
            sys.executable, "scripts/generate_figures.py",
            "--results-file", str(aggregated_results_file),
            "--output-dir", str(output_dir / "figures"),
            "--format", args.figure_format
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
    exit(main()) 