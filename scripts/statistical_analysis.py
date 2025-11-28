#!/usr/bin/env python3
"""
Statistical significance testing for benchmark results.

This script performs pairwise statistical tests between different approaches
to determine if performance differences are statistically significant.
"""

import argparse
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon


@dataclass
class StatisticalTest:
    """Container for statistical test results."""

    approach_a: str
    approach_b: str
    task: str
    metric: str
    test_type: str
    statistic: float
    p_value: float
    effect_size: float | None
    significant: bool
    sample_size: int


def load_sample_level_data(results_dir: Path) -> dict[str, dict[str, list[dict[str, float]]]]:
    """
    Load individual sample metrics for statistical testing.

    Returns:
        Dict[approach][task] -> List of individual sample metrics
    """
    sample_data = defaultdict(lambda: defaultdict(list))

    for approach_dir in results_dir.iterdir():
        if not approach_dir.is_dir():
            continue

        approach = approach_dir.name
        logger.info(f"Loading sample data for approach: {approach}")

        for task_dir in approach_dir.iterdir():
            if not task_dir.is_dir():
                continue

            task = task_dir.name

            for model_dir in task_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                # Find all metrics files for this approach/task/model combination
                metrics_files = list(model_dir.glob("*/image_*/metrics.json"))

                for metrics_file in metrics_files:
                    try:
                        with open(metrics_file) as f:
                            metrics = json.load(f)
                            sample_data[approach][task].append(metrics)
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        logger.warning(f"Failed to load {metrics_file}: {e}")

    return sample_data


def extract_metric_values(
    sample_data: dict[str, dict[str, list[dict[str, float]]]], approach: str, task: str, metric: str
) -> list[float]:
    """Extract values for a specific metric from sample data."""
    return [
        sample[metric] for sample in sample_data.get(approach, {}).get(task, []) if metric in sample
    ]


def compute_effect_size_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size for paired samples."""
    diff = a - b
    pooled_std = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    return np.mean(diff) / pooled_std if pooled_std > 0 else 0.0


def perform_paired_ttest(
    values_a: list[float],
    values_b: list[float],
    approach_a: str,
    approach_b: str,
    task: str,
    metric: str,
) -> StatisticalTest | None:
    """Perform paired t-test between two approaches."""
    if len(values_a) == 0 or len(values_b) == 0:
        return None

    # Find common samples (same indices)
    min_len = min(len(values_a), len(values_b))
    if min_len < 3:  # Need at least 3 samples for meaningful test
        return None

    a_array = np.array(values_a[:min_len])
    b_array = np.array(values_b[:min_len])

    # Perform paired t-test
    statistic, p_value = ttest_rel(a_array, b_array)

    # Compute effect size (Cohen's d)
    effect_size = compute_effect_size_cohens_d(a_array, b_array)

    return StatisticalTest(
        approach_a=approach_a,
        approach_b=approach_b,
        task=task,
        metric=metric,
        test_type="paired_t_test",
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        significant=p_value < 0.05,
        sample_size=min_len,
    )


def perform_wilcoxon_test(
    values_a: list[float],
    values_b: list[float],
    approach_a: str,
    approach_b: str,
    task: str,
    metric: str,
) -> StatisticalTest | None:
    """Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test)."""
    if len(values_a) == 0 or len(values_b) == 0:
        return None

    min_len = min(len(values_a), len(values_b))
    if min_len < 3:
        return None

    a_array = np.array(values_a[:min_len])
    b_array = np.array(values_b[:min_len])

    # Perform Wilcoxon signed-rank test
    try:
        statistic, p_value = wilcoxon(a_array, b_array, alternative="two-sided")
    except ValueError:
        # Handle case where all differences are zero
        return None

    # Effect size: r = Z / sqrt(N) where Z is the test statistic
    effect_size = abs(statistic) / np.sqrt(min_len)

    return StatisticalTest(
        approach_a=approach_a,
        approach_b=approach_b,
        task=task,
        metric=metric,
        test_type="wilcoxon_signed_rank",
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        significant=p_value < 0.05,
        sample_size=min_len,
    )


def perform_mcnemar_test(
    values_a: list[float],
    values_b: list[float],
    approach_a: str,
    approach_b: str,
    task: str,
    metric: str,
) -> StatisticalTest | None:
    """Perform McNemar's test for binary classification metrics."""
    if len(values_a) == 0 or len(values_b) == 0:
        return None

    min_len = min(len(values_a), len(values_b))
    if min_len < 3:
        return None

    a_array = np.array(values_a[:min_len])
    b_array = np.array(values_b[:min_len])

    # Check if values are binary (0/1 or boolean-like)
    if not (np.all(np.isin(a_array, [0, 1])) and np.all(np.isin(b_array, [0, 1]))):
        return None

    # Create contingency table for McNemar's test
    # Table structure: [[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]]
    np.sum((a_array == 1) & (b_array == 1))
    a_correct_b_wrong = np.sum((a_array == 1) & (b_array == 0))
    a_wrong_b_correct = np.sum((a_array == 0) & (b_array == 1))
    np.sum((a_array == 0) & (b_array == 0))

    # McNemar's test focuses on discordant pairs
    n_discordant = a_correct_b_wrong + a_wrong_b_correct

    if n_discordant == 0:
        return None  # No disagreement between models

    # McNemar's test statistic
    statistic = ((abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2) / n_discordant
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    # Effect size: Odds ratio
    if a_wrong_b_correct == 0:
        effect_size = float("inf") if a_correct_b_wrong > 0 else 1.0
    else:
        effect_size = a_correct_b_wrong / a_wrong_b_correct

    return StatisticalTest(
        approach_a=approach_a,
        approach_b=approach_b,
        task=task,
        metric=metric,
        test_type="mcnemar",
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        significant=p_value < 0.05,
        sample_size=min_len,
    )


def get_task_metrics(task: str) -> list[str]:
    """Get relevant metrics for statistical testing by task."""
    if task == "localization":
        return ["detection_mAP30", "detection_mAP50", "detection_mAP50_95"]
    elif task == "caption":
        return ["caption_bleu", "caption_meteor", "caption_bert_f1", "caption_radgraph_f1"]
    elif task == "diagnosis":
        return ["diagnosis_top1", "diagnosis_top5"]
    else:
        return []


def is_binary_metric(metric: str) -> bool:
    """Determine if a metric is binary for McNemar's test."""
    binary_metrics = ["diagnosis_top1", "diagnosis_top5"]
    return metric in binary_metrics


def perform_all_statistical_tests(
    sample_data: dict[str, dict[str, list[dict[str, float]]]],
) -> list[StatisticalTest]:
    """Perform all pairwise statistical tests between approaches."""
    results = []

    approaches = list(sample_data.keys())
    tasks = set()
    for approach_data in sample_data.values():
        tasks.update(approach_data.keys())

    logger.info(
        f"Performing statistical tests for {len(approaches)} approaches and {len(tasks)} tasks"
    )

    for task in tasks:
        metrics = get_task_metrics(task)
        if not metrics:
            continue

        logger.info(f"Testing task: {task} with metrics: {metrics}")

        # Get approaches that have data for this task
        task_approaches = [a for a in approaches if task in sample_data[a] and sample_data[a][task]]

        # Perform pairwise tests
        for approach_a, approach_b in itertools.combinations(task_approaches, 2):
            logger.debug(f"Comparing {approach_a} vs {approach_b} for {task}")

            for metric in metrics:
                values_a = extract_metric_values(sample_data, approach_a, task, metric)
                values_b = extract_metric_values(sample_data, approach_b, task, metric)

                if len(values_a) == 0 or len(values_b) == 0:
                    continue

                # Choose appropriate test based on metric type
                if is_binary_metric(metric):
                    # McNemar's test for binary metrics
                    test_result = perform_mcnemar_test(
                        values_a, values_b, approach_a, approach_b, task, metric
                    )
                    if test_result:
                        results.append(test_result)
                else:
                    # Both parametric and non-parametric tests for continuous metrics
                    ttest_result = perform_paired_ttest(
                        values_a, values_b, approach_a, approach_b, task, metric
                    )
                    if ttest_result:
                        results.append(ttest_result)

                    wilcoxon_result = perform_wilcoxon_test(
                        values_a, values_b, approach_a, approach_b, task, metric
                    )
                    if wilcoxon_result:
                        results.append(wilcoxon_result)

    return results


def apply_multiple_testing_correction(
    results: list[StatisticalTest], method: str = "bonferroni"
) -> list[StatisticalTest]:
    """Apply multiple testing correction to p-values."""
    if not results:
        return results

    p_values = [r.p_value for r in results]

    if method == "bonferroni":
        # Bonferroni correction
        corrected_alpha = 0.05 / len(p_values)
        for result in results:
            result.significant = result.p_value < corrected_alpha

    elif method == "fdr":
        # False Discovery Rate (Benjamini-Hochberg)
        sorted_indices = np.argsort(p_values)
        corrected_significant = np.zeros(len(p_values), dtype=bool)

        for i, idx in enumerate(sorted_indices):
            critical_value = (i + 1) / len(p_values) * 0.05
            if p_values[idx] <= critical_value:
                corrected_significant[idx] = True
            else:
                break

        for i, result in enumerate(results):
            result.significant = corrected_significant[i]

    return results


def save_statistical_results(results: list[StatisticalTest], output_dir: Path) -> None:
    """Save statistical test results to various formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame for easy manipulation
    data = [
        {
            "approach_a": result.approach_a,
            "approach_b": result.approach_b,
            "task": result.task,
            "metric": result.metric,
            "test_type": result.test_type,
            "statistic": result.statistic,
            "p_value": result.p_value,
            "effect_size": result.effect_size,
            "significant": result.significant,
            "sample_size": result.sample_size,
        }
        for result in results
    ]

    df = pd.DataFrame(data)

    # Save as CSV
    df.to_csv(output_dir / "statistical_tests.csv", index=False)
    logger.info(f"Saved statistical test results to {output_dir / 'statistical_tests.csv'}")

    # Create summary of significant results
    significant_results = df[df["significant"]]
    significant_results.to_csv(output_dir / "significant_results.csv", index=False)
    logger.info(f"Saved significant results to {output_dir / 'significant_results.csv'}")

    # Generate summary statistics
    summary = {
        "total_tests": len(results),
        "significant_tests": len(significant_results),
        "significance_rate": len(significant_results) / len(results) if results else 0,
        "tests_by_type": df["test_type"].value_counts().to_dict(),
        "significant_by_task": significant_results["task"].value_counts().to_dict(),
        "significant_by_metric": significant_results["metric"].value_counts().to_dict(),
    }

    with open(output_dir / "statistical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved statistical summary to {output_dir / 'statistical_summary.json'}")


def generate_statistical_report(results: list[StatisticalTest], output_dir: Path) -> None:
    """Generate a comprehensive statistical analysis report."""
    significant_results = [r for r in results if r.significant]

    report_lines = [
        "# Statistical Significance Analysis Report",
        "",
        f"**Total tests performed:** {len(results)}",
        f"**Statistically significant results:** {len(significant_results)} ({len(significant_results) / len(results) * 100:.1f}%)",
        "",
        "## Summary of Significant Findings",
        "",
    ]

    # Group significant results by task
    by_task = defaultdict(list)
    for result in significant_results:
        by_task[result.task].append(result)

    for task, task_results in by_task.items():
        report_lines.append(f"### {task.title()} Task")
        report_lines.append("")

        for result in task_results:
            effect_desc = ""
            if result.effect_size is not None:
                if result.test_type == "mcnemar":
                    effect_desc = f" (Odds Ratio: {result.effect_size:.2f})"
                else:
                    effect_desc = f" (Effect Size: {result.effect_size:.3f})"

            comparison = f"**{result.approach_a}** vs **{result.approach_b}**"
            metric_name = result.metric.replace("_", " ").title()
            report_lines.append(
                f"- {comparison} on {metric_name}: p = {result.p_value:.4f}{effect_desc}"
            )

        report_lines.append("")

    # Add test interpretation guide
    report_lines.extend(
        [
            "## Statistical Test Interpretation Guide",
            "",
            "### Effect Size Interpretation (Cohen's d)",
            "- Small effect: d ≈ 0.2",
            "- Medium effect: d ≈ 0.5",
            "- Large effect: d ≈ 0.8",
            "",
            "### P-value Interpretation",
            "- p < 0.05: Statistically significant difference",
            "- p < 0.01: Highly significant difference",
            "- p < 0.001: Very highly significant difference",
            "",
            "### Test Types Used",
            "- **Paired t-test**: For continuous metrics (parametric)",
            "- **Wilcoxon signed-rank**: For continuous metrics (non-parametric)",
            "- **McNemar's test**: For binary classification metrics",
            "",
            "---",
            "",
            "*Statistical analysis performed with multiple testing correction*",
        ]
    )

    # Write report
    report_file = output_dir / "statistical_analysis_report.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Statistical analysis report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Perform statistical significance testing on benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="runs/full_benchmark",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/statistical_analysis",
        help="Directory to save statistical analysis results",
    )
    parser.add_argument(
        "--correction-method",
        choices=["bonferroni", "fdr", "none"],
        default="fdr",
        help="Multiple testing correction method",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if not args.verbose:
        logger.remove()
        logger.add(lambda _: None)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return 1

    logger.info(f"Loading sample-level data from: {results_dir}")

    # Load individual sample data
    sample_data = load_sample_level_data(results_dir)

    if not sample_data:
        logger.error("No sample data found!")
        return 1

    logger.info(f"Loaded data for {len(sample_data)} approaches")

    # Perform statistical tests
    logger.info("Performing statistical tests...")
    results = perform_all_statistical_tests(sample_data)

    if not results:
        logger.error("No statistical tests could be performed!")
        return 1

    logger.info(f"Performed {len(results)} statistical tests")

    # Apply multiple testing correction
    if args.correction_method != "none":
        logger.info(f"Applying {args.correction_method} correction for multiple testing")
        results = apply_multiple_testing_correction(results, args.correction_method)

    # Save results
    save_statistical_results(results, output_dir)
    generate_statistical_report(results, output_dir)

    significant_count = sum(1 for r in results if r.significant)
    logger.info(
        f"Statistical analysis completed: {significant_count}/{len(results)} tests significant"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
