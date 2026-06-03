"""Confidence calibration utilities for ablation study analysis.

Provides statistical tools for evaluating model confidence calibration,
reliability flag analysis, and uncertainty quantification.
Integrates with existing evaluation infrastructure while providing
research-grade calibration metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from beartype import beartype
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Default thresholds for calibration analysis
DEFAULT_N_BINS = 10
DEFAULT_MIN_CONFIDENCE = 0.7  # Default threshold for filtering reliable predictions


@beartype
def calculate_reliability_diagram_data(
    confidences: list[float] | np.ndarray,
    correct: list[bool] | np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, Any]:
    """Calculate reliability diagram data for confidence calibration analysis.

    Args:
        confidences: List/array of confidence scores (0.0-1.0)
        correct: List/array of correctness indicators (True/False)
        n_bins: Number of bins for reliability diagram

    Returns:
        Dictionary with reliability diagram data and metrics
    """
    # Convert to numpy arrays if needed
    confidences = np.array(confidences)
    correct = np.array(correct).astype(int)

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(correct, confidences, n_bins=n_bins)

    # Calculate Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        # Find predictions in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    # Additional metrics
    accuracy = correct.mean()
    avg_confidence = confidences.mean()
    calibration_error = abs(avg_confidence - accuracy)

    # Brier score
    brier_score = brier_score_loss(correct, confidences)

    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "bin_boundaries": bin_boundaries.tolist(),
        "n_bins": n_bins,
        "metrics": {
            "accuracy": float(accuracy),
            "avg_confidence": float(avg_confidence),
            "calibration_error": float(calibration_error),
            "expected_calibration_error": float(ece),
            "brier_score": float(brier_score),
        },
        "sample_count": len(confidences),
    }


@beartype
def analyze_confidence_levels(
    confidences: list[float] | np.ndarray,
    confidence_levels: list[str],
    correct: list[bool] | np.ndarray,
    level_order: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze performance by confidence level categories.

    Args:
        confidences: List/array of confidence scores
        confidence_levels: List of confidence level labels (definite, probable, possible, uncertain)
        correct: List/array of correctness indicators
        level_order: Optional ordering of confidence levels (defaults to standard order)

    Returns:
        Analysis results by confidence level
    """
    if level_order is None:
        level_order = ["definite", "probable", "possible", "uncertain"]

    # Convert to numpy arrays
    conf_array = np.array(confidences)
    levels_array = np.array(confidence_levels)
    correct_array = np.array(correct).astype(bool)

    results: dict[str, Any] = {}

    for level in level_order:
        level_mask = levels_array == level
        if not np.any(level_mask):
            continue

        level_confidences = conf_array[level_mask]
        level_correct = correct_array[level_mask]

        if len(level_confidences) == 0:
            continue

        # Calculate metrics for this level
        accuracy = level_correct.mean()
        avg_confidence = level_confidences.mean()
        calibration_error = abs(avg_confidence - accuracy)

        results[level] = {
            "count": int(np.sum(level_mask)),
            "accuracy": float(accuracy),
            "avg_confidence": float(avg_confidence),
            "calibration_error": float(calibration_error),
            "confidence_range": [
                float(np.min(level_confidences)),
                float(np.max(level_confidences)),
            ],
        }

    # Overall statistics
    overall_accuracy = correct_array.mean()
    overall_avg_confidence = conf_array.mean()

    results["overall"] = {
        "total_count": len(confidences),
        "accuracy": float(overall_accuracy),
        "avg_confidence": float(overall_avg_confidence),
        "overall_calibration_error": abs(overall_avg_confidence - overall_accuracy),
    }

    return results


@beartype
def analyze_reliability_flags(
    reliable_flags: list[bool] | np.ndarray,
    confidences: list[float] | np.ndarray,
    correct: list[bool] | np.ndarray,
) -> dict[str, Any]:
    """Analyze reliability flag performance and discrimination.

    Args:
        reliable_flags: List/array of reliability flag predictions
        confidences: List/array of confidence scores
        correct: List/array of correctness indicators

    Returns:
        Reliability flag analysis results
    """
    # Convert to numpy arrays
    reliable_flags = np.array(reliable_flags)
    confidences = np.array(confidences)
    correct = np.array(correct).astype(bool)

    # Separate predictions by reliability flag
    reliable_predictions = reliable_flags
    unreliable_predictions = ~reliable_flags

    # Calculate metrics for each group
    results = {}

    # Reliable predictions
    if np.any(reliable_predictions):
        reliable_confidences = confidences[reliable_predictions]
        reliable_correct = correct[reliable_predictions]

        results["reliable"] = {
            "count": int(np.sum(reliable_predictions)),
            "accuracy": float(reliable_correct.mean()),
            "avg_confidence": float(reliable_confidences.mean()),
            "confidence_std": float(reliable_confidences.std()),
        }

    # Unreliable predictions
    if np.any(unreliable_predictions):
        unreliable_confidences = confidences[unreliable_predictions]
        unreliable_correct = correct[unreliable_predictions]

        results["unreliable"] = {
            "count": int(np.sum(unreliable_predictions)),
            "accuracy": float(unreliable_correct.mean()),
            "avg_confidence": float(unreliable_confidences.mean()),
            "confidence_std": float(unreliable_confidences.std()),
        }

    # Discrimination metrics
    if "reliable" in results and "unreliable" in results:
        results["discrimination"] = {
            "accuracy_difference": results["reliable"]["accuracy"]
            - results["unreliable"]["accuracy"],
            "confidence_difference": results["reliable"]["avg_confidence"]
            - results["unreliable"]["avg_confidence"],
            "reliable_rate": results["reliable"]["count"] / len(correct),
        }

    # Overall metrics
    results["overall"] = {
        "total_count": len(correct),
        "accuracy": float(correct.mean()),
        "reliable_rate": float(np.mean(reliable_flags)),
    }

    return results


@beartype
def compare_calibration_across_configs(
    config_results: dict[str, dict[str, Any]],
    metric: str = "expected_calibration_error",
) -> pd.DataFrame:
    """Compare calibration metrics across different ablation configurations.

    Args:
        config_results: Dictionary mapping config names to their calibration analysis results
        metric: Which calibration metric to compare (ece, accuracy, etc.)

    Returns:
        DataFrame with comparison results
    """
    comparison_data = []

    for config_name, results in config_results.items():
        if "calibration_metrics" in results:
            metrics = results["calibration_metrics"]
        elif "metrics" in results:
            metrics = results["metrics"]
        else:
            continue

        # Extract the requested metric
        if metric == "ece" and "expected_calibration_error" in metrics:
            value = metrics["expected_calibration_error"]
        elif metric == "accuracy" and "accuracy" in metrics:
            value = metrics["accuracy"]
        elif metric == "brier_score" and "brier_score" in metrics:
            value = metrics["brier_score"]
        elif metric == "calibration_error" and "calibration_error" in metrics:
            value = metrics["calibration_error"]
        elif metric in metrics:
            value = metrics[metric]
        else:
            continue

        comparison_data.append(
            {
                "configuration": config_name,
                "metric": metric,
                "value": float(value),
                "sample_count": results.get(
                    "total_predictions", results.get("sample_count", "N/A")
                ),
            }
        )

    return pd.DataFrame(comparison_data)


@beartype
def load_calibration_data_from_files(
    results_dir: Path,
    pattern: str = "*_results.json",
) -> dict[str, Any]:
    """Load calibration data from multiple result files.

    Args:
        results_dir: Directory containing result files
        pattern: File pattern to match (glob-style)

    Returns:
        Dictionary of calibration data by configuration
    """
    calibration_data = {}

    for result_file in results_dir.glob(pattern):
        try:
            with open(result_file) as f:
                data = json.load(f)

            # Extract calibration-relevant data
            config_name = data.get("config_name", result_file.stem)
            confidence = data.get("confidence", 0.0)

            if "final_response" in data:
                final_response = data["final_response"]
                confidence_level = final_response.get("confidence_level", "unknown")
                # Extract reliable flag - None if not present (don't assume reliability)
                reliability_flag_data = final_response.get("reliability_flag", {})
                reliable_flag = (
                    reliability_flag_data.get("reliable") if reliability_flag_data else None
                )
            else:
                confidence_level = "unknown"
                reliable_flag = None  # Unknown reliability - don't assume True

            if config_name not in calibration_data:
                calibration_data[config_name] = {
                    "confidences": [],
                    "confidence_levels": [],
                    "reliable_flags": [],
                    "correct": [],  # To be filled later with ground truth
                    "sample_ids": [],
                }

            calibration_data[config_name]["confidences"].append(confidence)
            calibration_data[config_name]["confidence_levels"].append(confidence_level)
            calibration_data[config_name]["reliable_flags"].append(reliable_flag)
            calibration_data[config_name]["sample_ids"].append(
                data.get("item_id", result_file.stem)
            )

        except OSError as e:
            raise OSError(f"Failed to read result file {result_file}: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in result file {result_file}: {e}") from e

    return calibration_data


@beartype
def create_calibration_summary(
    calibration_data: dict[str, Any],
    ground_truth: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Create comprehensive calibration analysis summary.

    Args:
        calibration_data: Calibration data loaded from result files
        ground_truth: Optional ground truth correctness mapping

    Returns:
        Comprehensive calibration analysis summary
    """
    configurations: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, list[dict[str, Any]]] = {}
    overall_stats: dict[str, int] = {
        "total_configurations": len(calibration_data),
        "total_samples": sum(len(data["confidences"]) for data in calibration_data.values()),
    }
    summary: dict[str, Any] = {
        "configurations": configurations,
        "comparisons": comparisons,
        "overall_stats": overall_stats,
    }

    # Analyze each configuration
    for config_name, data in calibration_data.items():
        config_summary = {}

        # Basic calibration metrics
        if data["confidences"]:
            config_summary["sample_count"] = len(data["confidences"])
            config_summary["avg_confidence"] = np.mean(data["confidences"])
            config_summary["confidence_std"] = np.std(data["confidences"])

            # Confidence level analysis
            if data["confidence_levels"]:
                level_counts = {}
                for level in data["confidence_levels"]:
                    level_counts[level] = level_counts.get(level, 0) + 1
                config_summary["confidence_level_distribution"] = level_counts

            # Reliability flag analysis - filter out None values
            valid_reliable_flags = [f for f in data["reliable_flags"] if f is not None]
            if valid_reliable_flags:
                reliable_rate = np.mean(valid_reliable_flags)
                config_summary["reliable_flag_rate"] = float(reliable_rate)

            # Add ground truth analysis if available
            if ground_truth:
                correct_flags = []
                missing_ground_truth = []
                for sample_id in data["sample_ids"]:
                    if sample_id in ground_truth:
                        correct_flags.append(ground_truth[sample_id])
                    else:
                        missing_ground_truth.append(sample_id)

                if missing_ground_truth:
                    raise ValueError(
                        f"Missing ground truth for {len(missing_ground_truth)} samples in "
                        f"config '{config_name}': {missing_ground_truth[:5]}..."
                        if len(missing_ground_truth) > 5
                        else f"Missing ground truth for samples: {missing_ground_truth}"
                    )

                data["correct"] = correct_flags

                # Run full calibration analysis
                reliability_data = calculate_reliability_diagram_data(
                    data["confidences"], data["correct"]
                )
                config_summary["calibration_metrics"] = reliability_data["metrics"]

                if data["confidence_levels"]:
                    level_analysis = analyze_confidence_levels(
                        data["confidences"], data["confidence_levels"], data["correct"]
                    )
                    config_summary["confidence_level_analysis"] = level_analysis

                if data["reliable_flags"]:
                    flag_analysis = analyze_reliability_flags(
                        data["reliable_flags"], data["confidences"], data["correct"]
                    )
                    config_summary["reliability_flag_analysis"] = flag_analysis

        configurations[config_name] = config_summary

    # Create comparisons if ground truth is available
    if ground_truth and len(calibration_data) > 1:
        comparison_metrics = ["accuracy", "expected_calibration_error", "brier_score"]
        for metric in comparison_metrics:
            comparison_df = compare_calibration_across_configs(
                {
                    config: data["calibration_metrics"]
                    for config, data in configurations.items()
                    if "calibration_metrics" in data
                },
                metric,
            )
            comparisons[metric] = comparison_df.to_dict("records")

    return summary


# Utility functions for batch processing integration
@beartype
def add_calibration_metadata_to_batch(
    batch_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add calibration metadata to batch processing results.

    Args:
        batch_results: List of batch processing results

    Returns:
        Enhanced results with calibration metadata
    """
    enhanced_results = []

    for result in batch_results:
        enhanced_result = result.copy()

        # Extract calibration-relevant data
        if "final_response" in result:
            final_response = result["final_response"]
            reliability_flag_data = final_response.get("reliability_flag", {})
            enhanced_result["calibration_metadata"] = {
                "confidence": result.get("confidence", 0.0),
                "confidence_level": final_response.get("confidence_level", "unknown"),
                "reliable_flag": reliability_flag_data.get("reliable")
                if reliability_flag_data
                else None,
                "final_prediction": final_response.get("final_prediction", {}),
                "research_metrics": result.get("research_metrics", {}),
            }

        enhanced_results.append(enhanced_result)

    return enhanced_results


@beartype
def filter_by_reliability_threshold(
    results: list[dict[str, Any]],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    only_reliable: bool = True,
) -> list[dict[str, Any]]:
    """Filter results based on confidence and reliability thresholds.

    Args:
        results: List of results with calibration metadata
        min_confidence: Minimum confidence threshold
        only_reliable: Whether to include only reliable predictions

    Returns:
        Filtered results
    """
    filtered_results = []

    for result in results:
        if "calibration_metadata" not in result:
            continue

        metadata = result["calibration_metadata"]
        confidence = metadata.get("confidence", 0.0)
        reliable_flag = metadata.get("reliable_flag")

        # Apply filters - treat None as unknown (not reliable if only_reliable is True)
        meets_confidence = confidence >= min_confidence
        meets_reliability = not only_reliable or reliable_flag is True
        if meets_confidence and meets_reliability:
            filtered_results.append(result)

    return filtered_results


@beartype
def export_calibration_results(
    calibration_summary: dict[str, Any],
    output_path: Path,
    output_format: str = "json",
) -> None:
    """Export calibration analysis results.

    Args:
        calibration_summary: Calibration analysis summary
        output_path: Output file path
        output_format: Export format (json, csv)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format.lower() == "json":
        with open(output_path, "w") as f:
            json.dump(calibration_summary, f, indent=2, default=str)

    elif output_format.lower() == "csv":
        # Flatten the summary for CSV export
        flat_data = []
        for config_name, config_data in calibration_summary["configurations"].items():
            row = {"configuration": config_name}
            row.update(config_data)
            flat_data.append(row)

        df = pd.DataFrame(flat_data)
        df.to_csv(output_path, index=False)

    else:
        raise ValueError(f"Unsupported export format: {output_format}")

    logger.info(f"Calibration results exported to {output_path}")
