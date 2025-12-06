from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from loguru import logger
from PIL import Image
from PIL import ImageDraw

from src.utils.confidence_calibration_utils import calculate_reliability_diagram_data


def overlay_boxes(
    image_path: Path,
    boxes: list[list[float]],
    labels: list[str] | None = None,
    color: str = "red",
    width: int = 2,
) -> Image.Image:
    """
    Draw bounding boxes on an image and optionally labels.

    Args:
        image_path: Path to the image file.
        boxes: List of [x1, y1, x2, y2].
        labels: Optional list of labels for each box.
        color: Color for the boxes.
        width: Line width.
    Returns:
        PIL Image with boxes drawn.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if labels and idx < len(labels):
            draw.text((x1, y1 - 10), labels[idx], fill=color)
    return image


def plot_metrics(
    metrics: dict[str, float],
    out_file: Path,
    title: str = "Evaluation Metrics",
) -> None:
    """
    Create a bar chart of metrics and save to file.

    Args:
        metrics: Dict of metric name to value.
        out_file: Path to save the plot (PNG).
        title: Plot title.
    """
    names = list(metrics.keys())
    values = [metrics[k] for k in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, values, color="skyblue")
    plt.title(title)
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_overlays(
    run_dir: Path,
    out_dir: Path,
    sample_idx: int = 0,
) -> None:
    """
    Create side-by-side overlay of ground truth vs predicted boxes for one sample.

    Args:
        run_dir: Directory containing 'preds.jsonl'.
        out_dir: Directory to save overlay image.
        sample_idx: Index of the sample to visualize.
    """
    preds_file = run_dir / "preds.jsonl"
    refs_file = run_dir / "refs.jsonl"
    with open(preds_file) as f:
        preds = [json.loads(line) for line in f]
    with open(refs_file) as f:
        refs = [json.loads(line) for line in f]
    if sample_idx >= len(preds) or sample_idx >= len(refs):
        raise IndexError(f"sample_idx {sample_idx} out of range")
    pred = preds[sample_idx]
    ref = refs[sample_idx]
    image_path = Path(pred.get("image_path", ""))
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Draw ground truth and prediction
    gt_img = overlay_boxes(
        image_path,
        ref.get("boxes", []),
        labels=[str(label) for label in ref.get("labels", [])],
        color="green",
    )
    pred_img = overlay_boxes(
        image_path,
        pred.get("boxes", []),
        labels=[str(label) for label in pred.get("labels", [])],
        color="red",
    )

    # Combine side by side
    w, h = gt_img.size
    canvas = Image.new("RGB", (w * 2 + 10, h))
    canvas.paste(gt_img, (0, 0))
    canvas.paste(pred_img, (w + 10, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), "Ground Truth", fill="green")
    draw.text((w + 15, 5), "Prediction", fill="red")

    out_file = out_dir / f"overlay_{sample_idx}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_file)


def plot_reliability_diagram(
    confidences: list[float] | npt.NDArray[np.floating[Any]],
    correct: list[bool] | npt.NDArray[np.bool_],
    output_path: Path,
    title: str = "Confidence Calibration",
    n_bins: int = 10,
) -> None:
    """
    Create a reliability diagram for confidence calibration analysis.

    Args:
        confidences: List/array of confidence scores (0.0-1.0)
        correct: List/array of correctness indicators (True/False)
        output_path: Path to save the plot
        title: Plot title
        n_bins: Number of bins for calibration
    """
    # Calculate reliability diagram data
    reliability_data = calculate_reliability_diagram_data(confidences, correct, n_bins)

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Plot model calibration
    prob_true = reliability_data["prob_true"]
    prob_pred = reliability_data["prob_pred"]
    plt.plot(prob_pred, prob_true, "s-", linewidth=2, markersize=8, label="Model")

    # Add confidence intervals
    bin_boundaries = reliability_data["bin_boundaries"]
    for i in range(len(prob_pred)):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        plt.plot([bin_lower, bin_upper], [prob_true[i], prob_true[i]], "gray", alpha=0.3)

    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Fraction of positives")
    plt.title(f"{title}\n(ECE: {reliability_data['metrics']['expected_calibration_error']:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ablation_comparison(
    results: dict[str, dict[str, Any]],
    output_path: Path,
    metric: str = "accuracy",
    title: str = "Ablation Study Comparison",
) -> None:
    """
    Create comparison plot for ablation study results.

    Args:
        results: Dictionary mapping config names to their results
        output_path: Path to save the plot
        metric: Which metric to compare (accuracy, confidence, etc.)
        title: Plot title
    """
    # Extract data for plotting
    configs = []
    values = []
    errors = []

    for config_name, config_data in results.items():
        configs.append(config_name.replace("_", " ").title())

        if metric == "accuracy":
            value = config_data.get("accuracy", 0.0)
            errors.append(0.0)  # Add confidence intervals if available
        elif metric == "confidence":
            value = config_data.get("avg_confidence", 0.0)
            errors.append(0.0)
        elif metric == "tool_calls":
            research_metrics = config_data.get("research_metrics", {})
            value = research_metrics.get("total_tool_calls", 0)
            errors.append(0.0)
        else:
            value = config_data.get(metric, 0.0)
            errors.append(0.0)

        values.append(value)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Color bars by performance
    colors = plt.cm.RdYlBu_r(np.array(values))
    bars = plt.bar(configs, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1)

    # Add value labels on bars
    for _i, (bar, value) in enumerate(zip(bars, values, strict=False)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{value:.3f}" if isinstance(value, float) else f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.xlabel("Configuration")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confidence_level_analysis(
    confidence_levels: dict[str, dict[str, Any]],
    output_path: Path,
    title: str = "Confidence Level Analysis",
) -> None:
    """
    Create analysis plot for confidence level performance.

    Args:
        confidence_levels: Confidence level analysis results
        output_path: Path to save the plot
        title: Plot title
    """
    # Prepare data for plotting
    levels = []
    accuracies = []
    confidences = []
    counts = []

    for level, data in confidence_levels.items():
        if level == "overall":
            continue

        levels.append(level.title())
        accuracies.append(data["accuracy"])
        confidences.append(data["avg_confidence"])
        counts.append(data["count"])

    if not levels:
        return

    # Create the plot
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot comparing accuracy vs confidence
    x = np.arange(len(levels))
    width = 0.35

    ax1.bar(x - width / 2, accuracies, width, label="Actual Accuracy", alpha=0.8, color="skyblue")
    ax1.bar(
        x + width / 2,
        confidences,
        width,
        label="Predicted Confidence",
        alpha=0.8,
        color="lightcoral",
    )

    ax1.set_xlabel("Confidence Level")
    ax1.set_ylabel("Score")
    ax1.set_title("Accuracy vs Predicted Confidence")
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add sample counts as text on bars
    for i, count in enumerate(counts):
        ax1.text(
            i,
            max(accuracies[i], confidences[i]) + 0.02,
            f"n={count}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Calibration error plot
    calibration_errors = [
        abs(acc - conf) for acc, conf in zip(accuracies, confidences, strict=False)
    ]
    bars = ax2.bar(levels, calibration_errors, color="orange", alpha=0.7)
    ax2.set_xlabel("Confidence Level")
    ax2.set_ylabel("Calibration Error")
    ax2.set_title("Calibration Error by Level")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, error in zip(bars, calibration_errors, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{error:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.suptitle(title)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_reliability_flag_analysis(
    reliability_data: dict[str, dict[str, Any]],
    output_path: Path,
    title: str = "Reliability Flag Analysis",
) -> None:
    """
    Create analysis plot for reliability flag performance.

    Args:
        reliability_data: Reliability flag analysis results
        output_path: Path to save the plot
        title: Plot title
    """
    if "reliable" not in reliability_data or "unreliable" not in reliability_data:
        logger.warning("Missing reliability data for plotting - skipping reliability flag analysis")
        return

    reliable = reliability_data["reliable"]
    unreliable = reliability_data["unreliable"]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy comparison
    categories = ["Flagged as\nReliable", "Flagged as\nUnreliable"]
    accuracies = [reliable["accuracy"], unreliable["accuracy"]]
    colors = ["green", "orange"]

    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.7)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy by Reliability Flag")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels and sample counts
    for i, (bar, acc, _cat) in enumerate(zip(bars, accuracies, categories, strict=False)):
        count = reliable["count"] if i == 0 else unreliable["count"]
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.3f}\n(n={count})",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Confidence distribution
    confidences = [reliable["avg_confidence"], unreliable["avg_confidence"]]
    confidence_stds = [reliable["confidence_std"], unreliable["confidence_std"]]

    bars2 = ax2.bar(
        categories, confidences, yerr=confidence_stds, color=colors, alpha=0.7, capsize=10
    )
    ax2.set_ylabel("Average Confidence")
    ax2.set_title("Confidence Distribution by Flag")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for _i, (bar, conf) in enumerate(zip(bars2, confidences, strict=False)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{conf:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add discrimination info if available
    if "discrimination" in reliability_data:
        disc = reliability_data["discrimination"]
        discrimination = disc["accuracy_difference"]
        fig.text(
            0.5,
            0.02,
            f"Discrimination Score: {discrimination:.3f}",
            ha="center",
            fontsize=12,
            fontweight="bold",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

    plt.suptitle(title)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tool_usage_heatmap(
    tool_usage_data: dict[str, dict[str, int]],
    output_path: Path,
    title: str = "Tool Usage Heatmap",
) -> None:
    """
    Create heatmap visualization of tool usage across configurations.

    Args:
        tool_usage_data: Dictionary mapping configs to tool usage counts
        output_path: Path to save the plot
        title: Plot title
    """
    # Convert to DataFrame
    df_data = []
    configs = []
    all_tools = set()

    for config, usage in tool_usage_data.items():
        configs.append(config.replace("_", " ").title())
        for tool, count in usage.items():
            all_tools.add(tool)
            df_data.append(
                {
                    "Configuration": config.replace("_", " ").title(),
                    "Tool": tool.replace("_", " ").title(),
                    "Usage": count,
                }
            )

    if not df_data:
        logger.info("No tool usage data available for plotting - skipping heatmap")
        return

    df = pd.DataFrame(df_data)

    # Create pivot table for heatmap
    pivot_df = df.pivot_table(index="Configuration", columns="Tool", values="Usage", fill_value=0)

    # Create the heatmap
    plt.figure(figsize=(14, max(8, len(configs) * 0.6)))

    sns.heatmap(pivot_df, annot=True, fmt="g", cmap="Blues", cbar_kws={"label": "Usage Count"})
    plt.title(title)
    plt.xlabel("Tools")
    plt.ylabel("Configurations")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_token_efficiency_scatter(
    results: dict[str, dict[str, Any]],
    output_path: Path,
    title: str = "Token Efficiency vs Performance",
) -> None:
    """
    Create scatter plot of token efficiency vs performance.

    Args:
        results: Results dictionary with token and performance data
        output_path: Path to save the plot
        title: Plot title
    """
    configs = []
    tokens = []
    confidences = []
    accuracies = []
    tool_calls = []

    for config_name, config_data in results.items():
        configs.append(config_name.replace("_", " ").title())

        # Extract metrics
        research_metrics = config_data.get("research_metrics", {})
        tokens.append(config_data.get("avg_tokens", research_metrics.get("avg_tokens", 0)))
        confidences.append(
            config_data.get("confidence", research_metrics.get("final_confidence_score", 0))
        )
        accuracies.append(config_data.get("accuracy", 0))
        tool_calls.append(research_metrics.get("total_tool_calls", 0))

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Scatter plot with tool calls as bubble size
    scatter = plt.scatter(
        tokens,
        accuracies,
        s=[tc * 50 + 100 for tc in tool_calls],
        alpha=0.6,
        c=confidences,
        cmap="RdYlBu_r",
        edgecolors="black",
        linewidth=1,
    )

    # Add labels for each point
    for i, config in enumerate(configs):
        plt.annotate(
            config,
            (tokens[i], accuracies[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    plt.xlabel("Average Tokens Used")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.colorbar(scatter, label="Confidence Score")
    plt.grid(True, alpha=0.3)

    # Add trend line
    if len(tokens) > 1:
        z = np.polyfit(tokens, accuracies, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(tokens), max(tokens), 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label="Trend line")
        plt.legend()

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_comprehensive_ablation_plots(
    results: dict[str, dict[str, Any]],
    output_dir: Path,
    study_name: str = "Ablation Study",
) -> dict[str, Path]:
    """
    Create all standard ablation study visualization plots.

    Args:
        results: Ablation study results
        output_dir: Directory to save all plots
        study_name: Name of the study for plot titles

    Returns:
        Dictionary mapping plot types to their file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    # 1. Performance comparison
    perf_path = output_dir / f"{study_name}_performance_comparison.png"
    plot_ablation_comparison(
        results, perf_path, metric="accuracy", title=f"{study_name}: Performance Comparison"
    )
    plot_paths["performance_comparison"] = perf_path

    # 2. Confidence comparison
    conf_path = output_dir / f"{study_name}_confidence_comparison.png"
    plot_ablation_comparison(
        results, conf_path, metric="confidence", title=f"{study_name}: Confidence Comparison"
    )
    plot_paths["confidence_comparison"] = conf_path

    # 3. Tool usage heatmap (if available)
    tool_usage_data = {}
    for config_name, config_data in results.items():
        research_metrics = config_data.get("research_metrics", {})
        if "tools_usage_breakdown" in research_metrics:
            tool_usage_data[config_name] = research_metrics["tools_usage_breakdown"]

    if tool_usage_data:
        tool_path = output_dir / f"{study_name}_tool_usage_heatmap.png"
        plot_tool_usage_heatmap(
            tool_usage_data, tool_path, title=f"{study_name}: Tool Usage Patterns"
        )
        plot_paths["tool_usage_heatmap"] = tool_path

    # 4. Token efficiency scatter
    token_path = output_dir / f"{study_name}_token_efficiency.png"
    plot_token_efficiency_scatter(
        results, token_path, title=f"{study_name}: Token Efficiency vs Performance"
    )
    plot_paths["token_efficiency"] = token_path

    # 5. Calibration plots (if calibration data available)
    calibration_configs = {}
    for config_name, config_data in results.items():
        if "calibration_metrics" in config_data:
            calibration_configs[config_name] = config_data["calibration_metrics"]

    if calibration_configs:
        # Reliability diagram for best performing config
        best_config = max(results.items(), key=lambda x: x[1].get("accuracy", 0))
        best_name, best_data = best_config

        if "confidences" in best_data and "correct" in best_data:
            reliability_path = output_dir / f"{study_name}_reliability_diagram.png"
            plot_reliability_diagram(
                best_data["confidences"],
                best_data["correct"],
                reliability_path,
                title=f"{study_name}: Reliability Diagram ({best_name})",
            )
            plot_paths["reliability_diagram"] = reliability_path

        # Confidence level analysis
        if "confidence_level_analysis" in best_data:
            level_path = output_dir / f"{study_name}_confidence_levels.png"
            plot_confidence_level_analysis(
                best_data["confidence_level_analysis"],
                level_path,
                title=f"{study_name}: Confidence Level Analysis ({best_name})",
            )
            plot_paths["confidence_levels"] = level_path

        # Reliability flag analysis
        if "reliability_flag_analysis" in best_data:
            flag_path = output_dir / f"{study_name}_reliability_flags.png"
            plot_reliability_flag_analysis(
                best_data["reliability_flag_analysis"],
                flag_path,
                title=f"{study_name}: Reliability Flag Analysis ({best_name})",
            )
            plot_paths["reliability_flags"] = flag_path

    return plot_paths
