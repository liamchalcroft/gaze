"""Statistical analysis utilities for ablation study results.

Provides comprehensive statistical analysis tools for evaluating
ablation study results, including significance testing, effect sizes,
and confidence intervals for research paper preparation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from beartype import beartype
from loguru import logger
from scipy import stats


@beartype
def calculate_effect_size(
    group1: list[float] | np.ndarray,
    group2: list[float] | np.ndarray,
    effect_type: str = "cohens_d",
) -> dict[str, float]:
    """Calculate effect size between two groups.

    Args:
        group1: First group of values
        group2: Second group of values
        effect_type: Type of effect size ('cohens_d', 'hedges_g', 'glass_delta')

    Returns:
        Dictionary with effect size and confidence intervals
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) == 0 or len(group2) == 0:
        return {"effect_size": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return {"effect_size": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    if effect_type == "cohens_d":
        # Cohen's d
        effect_size = (mean1 - mean2) / pooled_std
    elif effect_type == "hedges_g":
        # Hedges' g (bias-corrected Cohen's d)
        effect_size = (mean1 - mean2) / pooled_std
        # Bias correction
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
        effect_size *= correction_factor
    elif effect_type == "glass_delta":
        # Glass's Delta (uses control group std)
        effect_size = (mean1 - mean2) / std2
    else:
        raise ValueError(f"Unknown effect type: {effect_type}")

    # Calculate confidence intervals
    se = pooled_std * np.sqrt(1 / n1 + 1 / n2)
    ci_lower = effect_size - 1.96 * se
    ci_upper = effect_size + 1.96 * se

    return {
        "effect_size": float(effect_size),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n1": n1,
        "n2": n2,
        "mean1": float(mean1),
        "mean2": float(mean2),
    }


@beartype
def perform_statistical_test(
    group1: list[float] | np.ndarray,
    group2: list[float] | np.ndarray,
    test_type: str = "ttest",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Perform statistical test between two groups.

    Args:
        group1: First group of values
        group2: Second group of values
        test_type: Type of statistical test ('ttest', 'mannwhitney', 'wilcoxon')
        alpha: Significance level
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

    Returns:
        Dictionary with test statistics and p-values
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) == 0 or len(group2) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    result = {"n1": len(group1), "n2": len(group2), "alpha": alpha}

    if test_type == "ttest":
        # Student's t-test
        statistic, p_value = stats.ttest_ind(group1, group2, alternative=alternative)
        result["test"] = "Student's t-test"
        result["statistic"] = float(statistic)
        result["p_value"] = float(p_value)

    elif test_type == "mannwhitney":
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
        result["test"] = "Mann-Whitney U"
        result["statistic"] = float(statistic)
        result["p_value"] = float(p_value)

    elif test_type == "wilcoxon":
        # Wilcoxon signed-rank test (for paired data)
        if len(group1) != len(group2):
            raise ValueError("Wilcoxon test requires paired data of equal length")
        statistic, p_value = stats.wilcoxon(group1, group2, alternative=alternative)
        result["test"] = "Wilcoxon signed-rank"
        result["statistic"] = float(statistic)
        result["p_value"] = float(p_value)

    else:
        raise ValueError(f"Unknown test type: {test_type}")

    result["significant"] = p_value < alpha
    result["mean1"] = float(np.mean(group1))
    result["mean2"] = float(np.mean(group2))
    result["std1"] = float(np.std(group1, ddof=1))
    result["std2"] = float(np.std(group2, ddof=1))

    return result


@beartype
def analyze_ablation_configurations(
    results: dict[str, dict[str, Any]],
    baseline_config: str = "baseline_single_shot",
    metric: str = "accuracy",
) -> dict[str, Any]:
    """Perform statistical analysis of ablation configurations.

    Args:
        results: Dictionary mapping config names to their results
        baseline_config: Name of baseline configuration for comparison
        metric: Which metric to analyze

    Returns:
        Statistical analysis results
    """
    if baseline_config not in results:
        logger.warning(f"Baseline configuration {baseline_config} not found")
        return {}

    baseline_data = results[baseline_config]
    baseline_values = [baseline_data.get(metric, 0)]

    analysis_results = {
        "baseline_config": baseline_config,
        "baseline_value": baseline_values[0] if baseline_values else 0,
        "comparisons": {},
        "summary": {
            "total_configurations": len(results),
            "significant_improvements": 0,
            "significant_degradations": 0,
        },
    }

    for config_name, config_data in results.items():
        if config_name == baseline_config:
            continue

        config_values = [config_data.get(metric, 0)]

        # Perform statistical test
        test_result = perform_statistical_test(config_values, baseline_values, test_type="ttest")

        # Calculate effect size
        effect_result = calculate_effect_size(
            config_values, baseline_values, effect_type="cohens_d"
        )

        comparison_result = {
            "config_name": config_name,
            "config_value": config_values[0] if config_values else 0,
            "difference": config_values[0] - baseline_values[0] if config_values else 0,
            "relative_improvement": (
                (config_values[0] - baseline_values[0]) / baseline_values[0] * 100
            )
            if baseline_values[0] != 0
            else 0,
        }

        comparison_result.update(test_result)
        comparison_result.update(effect_result)

        analysis_results["comparisons"][config_name] = comparison_result

        # Update summary
        if test_result["significant"]:
            if comparison_result["difference"] > 0:
                analysis_results["summary"]["significant_improvements"] += 1
            else:
                analysis_results["summary"]["significant_degradations"] += 1

    return analysis_results


@beartype
def create_statistical_summary_table(
    results: dict[str, dict[str, Any]],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Create comprehensive statistical summary table.

    Args:
        results: Dictionary mapping config names to their results
        metrics: List of metrics to include (default: common metrics)

    Returns:
        DataFrame with statistical summary
    """
    if metrics is None:
        metrics = ["accuracy", "confidence", "avg_tokens", "total_tool_calls", "uncertainty_rate"]

    summary_data = []

    for config_name, config_data in results.items():
        row = {"Configuration": config_name.replace("_", " ").title()}

        for metric in metrics:
            if metric in config_data:
                row[metric.replace("_", " ").title()] = config_data[metric]
            else:
                # Try to extract from research metrics
                research_metrics = config_data.get("research_metrics", {})
                if metric == "total_tool_calls":
                    row["Tool Calls"] = research_metrics.get("total_tool_calls", 0)
                elif metric == "avg_tokens":
                    row["Avg Tokens"] = research_metrics.get("avg_tokens", 0)
                elif metric == "uncertainty_rate":
                    row["Uncertainty Rate"] = research_metrics.get("uncertainty_rate", 0)
                else:
                    row[metric.replace("_", " ").title()] = 0

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Sort by accuracy if available
    if "Accuracy" in df.columns:
        df = df.sort_values("Accuracy", ascending=False)

    return df


@beartype
def calculate_confidence_intervals(
    values: list[float] | np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Calculate confidence intervals for a set of values.

    Args:
        values: List/array of values
        confidence_level: Confidence level (0-1)

    Returns:
        Dictionary with confidence interval statistics
    """
    values = np.array(values)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0, "std": 0, "n": 0}

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
    margin_error = t_critical * (std / np.sqrt(n))

    ci_lower = mean - margin_error
    ci_upper = mean + margin_error

    return {
        "mean": float(mean),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std": float(std),
        "n": n,
        "confidence_level": confidence_level,
    }


@beartype
def analyze_calibration_reliability(
    calibration_data: dict[str, Any],
) -> dict[str, Any]:
    """Analyze calibration reliability and discrimination metrics.

    Args:
        calibration_data: Calibration analysis results

    Returns:
        Reliability analysis results
    """
    reliability_analysis = {}

    for config_name, config_data in calibration_data.items():
        if "calibration_metrics" not in config_data:
            continue

        metrics = config_data["calibration_metrics"]

        analysis = {
            "config_name": config_name,
            "expected_calibration_error": metrics.get("expected_calibration_error", 0),
            "brier_score": metrics.get("brier_score", 0),
            "overall_accuracy": metrics.get("overall_accuracy", 0),
        }

        # Reliability flag analysis
        if "reliability_analysis" in metrics:
            rel_analysis = metrics["reliability_analysis"]
            analysis["reliable_accuracy"] = rel_analysis.get("reliable_accuracy", 0)
            analysis["unreliable_accuracy"] = rel_analysis.get("unreliable_accuracy", 0)
            analysis["reliability_discrimination"] = rel_analysis.get(
                "reliability_discrimination", 0
            )
            analysis["reliable_rate"] = rel_analysis.get("reliability_flag_rate", 0)

        # Confidence level analysis
        if "level_breakdown" in metrics:
            level_data = metrics["level_breakdown"]
            analysis["confidence_level_calibration"] = {}
            for level, level_stats in level_data.items():
                analysis["confidence_level_calibration"][level] = {
                    "count": level_stats.get("count", 0),
                    "accuracy": level_stats.get("accuracy", 0),
                    "calibration_error": level_stats.get("calibration_error", 0),
                }

        reliability_analysis[config_name] = analysis

    return reliability_analysis


@beartype
def generate_paper_ready_statistics(
    results: dict[str, dict[str, Any]],
    output_path: Path,
    baseline_config: str = "baseline_single_shot",
) -> None:
    """Generate paper-ready statistical analysis.

    Args:
        results: Ablation study results
        output_path: Path to save statistics
        baseline_config: Baseline configuration name
    """
    # Create comprehensive analysis
    statistical_results = {
        "study_metadata": {
            "total_configurations": len(results),
            "baseline_configuration": baseline_config,
            "analysis_date": pd.Timestamp.now().isoformat(),
        },
        "summary_table": {},
        "statistical_tests": {},
        "effect_sizes": {},
        "calibration_analysis": {},
    }

    # 1. Summary statistics table
    summary_df = create_statistical_summary_table(results)
    statistical_results["summary_table"] = summary_df.to_dict("records")

    # 2. Statistical tests vs baseline
    if baseline_config in results:
        for metric in ["accuracy", "confidence", "avg_tokens"]:
            try:
                test_results = analyze_ablation_configurations(results, baseline_config, metric)
                statistical_results["statistical_tests"][metric] = test_results
            except Exception as e:
                logger.warning(f"Failed to analyze {metric}: {e}")

    # 3. Calibration reliability analysis
    calibration_data = {
        config_name: config_data
        for config_name, config_data in results.items()
        if "calibration_metrics" in config_data or "research_metrics" in config_data
    }

    if calibration_data:
        reliability_results = analyze_calibration_reliability(calibration_data)
        statistical_results["calibration_analysis"] = reliability_results

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(statistical_results, f, indent=2, default=str)

    # Save as CSV for easy spreadsheet import
    csv_path = output_path.with_suffix(".csv")
    summary_df.to_csv(csv_path, index=False)

    logger.info(f"Statistical analysis saved to {output_path}")


@beartype
def calculate_statistical_power(
    effect_size: float,
    sample_size: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Calculate achieved statistical power or required sample size.

    Args:
        effect_size: Effect size (Cohen's d)
        sample_size: Sample size per group
        alpha: Significance level
        power: Desired power

    Returns:
        Statistical power if sample_size provided
    """
    # Use normal approximation for power calculation
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    stats.norm.ppf(power)

    # Calculate non-centrality parameter
    ncp = effect_size * np.sqrt(sample_size / 2)

    # Calculate power
    power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)

    return float(power)


@beartype
def perform_multiple_comparison_correction(
    p_values: list[float],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> list[bool]:
    """Apply multiple comparison correction to p-values.

    Args:
        p_values: List of p-values
        method: Correction method ('bonferroni', 'fdr_bh')
        alpha: Original significance level

    Returns:
        List of corrected significance results
    """
    p_values = np.array(p_values)

    if method == "bonferroni":
        # Bonferroni correction
        corrected_alpha = alpha / len(p_values)
        significant = p_values < corrected_alpha
        return significant.tolist()

    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR correction
        from statsmodels.stats.multitest import multipletests

        rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
        return rejected.tolist()

    else:
        raise ValueError(f"Unknown correction method: {method}")


# Utility functions for batch processing integration
@beartype
def add_statistical_annotations(
    results: dict[str, dict[str, Any]],
    baseline_config: str = "baseline_single_shot",
) -> dict[str, dict[str, Any]]:
    """Add statistical annotations to results for paper preparation.

    Args:
        results: Original results dictionary
        baseline_config: Baseline configuration

    Returns:
        Enhanced results with statistical annotations
    """
    enhanced_results = results.copy()

    for config_name, config_data in enhanced_results.items():
        if config_name == baseline_config:
            config_data["statistical_note"] = "Baseline configuration"
            continue

        # Calculate improvement relative to baseline
        if baseline_config in results:
            baseline_data = results[baseline_config]

            for metric in ["accuracy", "confidence"]:
                baseline_val = baseline_data.get(metric, 0)
                current_val = config_data.get(metric, 0)

                if baseline_val > 0:
                    improvement = ((current_val - baseline_val) / baseline_val) * 100
                    config_data[f"{metric}_improvement_pct"] = improvement

                    # Add significance annotation
                    if improvement > 5:
                        config_data[f"{metric}_significance"] = "Substantial improvement"
                    elif improvement > 0:
                        config_data[f"{metric}_significance"] = "Modest improvement"
                    elif improvement < -5:
                        config_data[f"{metric}_significance"] = "Significant degradation"
                    else:
                        config_data[f"{metric}_significance"] = "No significant change"

    return enhanced_results
