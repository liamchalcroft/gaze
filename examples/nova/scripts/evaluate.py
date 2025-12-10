#!/usr/bin/env python3
"""NOVA Evaluation Script using Complete Dataset.

Evaluates per-subject predictions against complete NOVA dataset
(images + metadata + ground truth). This uses the proper HuggingFace
images combined with CSV ground truth and patient history.

Usage:
    python scripts/evaluate.py --results-dir ./results/baseline --output ./eval
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pandas as pd
from beartype import beartype
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.nova_dataset import NovaDataset
from src.evaluation.caption import evaluate_caption
from src.evaluation.detection import evaluate_detection
from src.evaluation.diagnosis import evaluate_diagnosis_nova_official


@beartype
def load_predictions(results_dir: Path) -> dict[int, dict[str, Any]]:
    """Load predictions from results directory with per-subject structure."""
    predictions = {}
    per_subject_dir = results_dir / "per_subject"

    if not per_subject_dir.exists():
        raise FileNotFoundError(f"Per-subject directory not found: {per_subject_dir}")

    subject_dirs = [d for d in per_subject_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(subject_dirs)} subject directories in {per_subject_dir}")

    for subject_dir in subject_dirs:
        try:
            # Handle both string and numeric subject IDs
            subject_name = subject_dir.name
            if subject_name.startswith("subject_"):
                subject_id = int(subject_name.replace("subject_", ""))
            else:
                subject_id = int(subject_name)

            prediction_file = subject_dir / "predictions.json"

            if prediction_file.exists():
                with open(prediction_file) as f:
                    pred_data = json.load(f)
                predictions[subject_id] = pred_data
            else:
                logger.warning(f"No prediction file found for subject {subject_name}")

        except ValueError:
            logger.warning(f"Skipping invalid subject directory: {subject_dir.name}")
        except Exception as e:
            logger.error(f"Error loading prediction from {subject_dir}: {e}")

    logger.info(f"Loaded {len(predictions)} subject predictions")
    return predictions


@beartype
def evaluate_caption_metrics(
    predictions: dict[int, dict[str, Any]], dataset: NovaDataset
) -> dict[str, Any]:
    """Evaluate caption predictions using NOVA caption evaluation metrics."""
    logger.info("📝 Evaluating caption metrics with complete dataset...")

    # Extract predicted and reference captions
    pred_captions = []
    ref_captions = []
    matched_subjects = []

    for subject_id, pred in predictions.items():
        try:
            # Get prediction
            caption_data = pred.get("caption", {})
            pred_caption = caption_data.get("description", "")

            if not pred_caption:
                continue

            # Get corresponding ground truth from complete dataset
            if subject_id < len(dataset):
                sample = dataset[subject_id]
                ref_caption = sample["ground_truth"]["caption"]
                if ref_caption:
                    pred_captions.append(pred_caption)
                    ref_captions.append(ref_caption)
                    matched_subjects.append(subject_id)
                    logger.debug(
                        f"Subject {subject_id}: GT file {sample['ground_truth']['filename']}"
                    )

        except Exception as e:
            logger.warning(f"Error processing caption for subject {subject_id}: {e}")

    if not pred_captions:
        logger.error("No valid caption pairs found for evaluation")
        return {"error": "No valid caption pairs found"}

    logger.info(f"Evaluating {len(pred_captions)} caption pairs")

    # Evaluate using NOVA caption metrics
    try:
        caption_scores = evaluate_caption(pred_captions, ref_captions)

        # Add additional metadata
        caption_metrics = {
            "total_subjects": len(predictions),
            "evaluated_subjects": len(matched_subjects),
            "matched_subjects": matched_subjects,
            "avg_caption_length": sum(len(c) for c in pred_captions) / len(pred_captions),
            **caption_scores,
        }

        logger.success(f"Caption evaluation completed: {len(matched_subjects)} subjects evaluated")
        return caption_metrics

    except Exception as e:
        logger.error(f"Caption evaluation failed: {e}")
        return {"error": str(e)}


@beartype
def evaluate_diagnosis_metrics(
    predictions: dict[int, dict[str, Any]],
    dataset: NovaDataset,
    semantic_model: str = "x-ai/grok-4.1-fast:free",
) -> dict[str, Any]:
    """Evaluate diagnosis predictions using NOVA diagnosis evaluation metrics."""
    logger.info("🩺 Evaluating diagnosis metrics with complete dataset...")

    # Extract predicted and reference diagnoses
    pred_diagnoses = []
    ref_diagnoses = []
    matched_subjects = []

    for subject_id, pred in predictions.items():
        try:
            # Get prediction
            diagnosis_data = pred.get("diagnosis", {})
            pred_diagnosis = diagnosis_data.get("primary_diagnosis", "")

            if not pred_diagnosis:
                continue

            # Extract core diagnosis from detailed prediction
            # Helps when model gives detailed descriptions like "SOD - ..."
            import re

            # Try to extract the main diagnosis term
            pred_diagnosis_clean = pred_diagnosis

            # Look for diagnosis patterns - more comprehensive
            # Pattern for common diagnosis suffixes
            suffix_pat = (
                r"^([A-Za-z\s\-]+(?:dysplasia|syndrome|disease|disorder|"
                r"malformation|agenesis|atrophy|stenosis|astrocytoma|glioma|"
                r"medulloblastoma|meningioma|hemorrhage|infarction|hematoma|cyst))"
            )
            # Pattern for named conditions
            named_pat = (
                r"^([A-Za-z\s\-]+(?:Chiari|Dandy-Walker|Moyamoya|"
                r"Multiple sclerosis|Arachnoid|Subdural|Intracerebral))"
            )
            patterns = [suffix_pat, named_pat, r"^([A-Za-z\s\-]+)"]

            for pattern in patterns:
                match = re.search(pattern, pred_diagnosis_clean)
                if match:
                    pred_diagnosis_clean = match.group(1).strip()
                    break

            # Get corresponding ground truth from complete dataset
            if subject_id < len(dataset):
                sample = dataset[subject_id]
                ref_diagnosis = sample["ground_truth"]["final_diagnosis"]
                if ref_diagnosis:
                    pred_diagnoses.append(pred_diagnosis_clean)
                    ref_diagnoses.append(ref_diagnosis)
                    matched_subjects.append(subject_id)

        except Exception as e:
            logger.warning(f"Error processing diagnosis for subject {subject_id}: {e}")

    if not pred_diagnoses:
        logger.error("No valid diagnosis pairs found for evaluation")
        return {"error": "No valid diagnosis pairs found"}

    logger.info(f"Evaluating {len(pred_diagnoses)} diagnosis pairs")

    # Evaluate using NOVA diagnosis metrics
    try:
        diagnosis_scores = evaluate_diagnosis_nova_official(
            pred_diagnoses, ref_diagnoses, model_name=semantic_model
        )

        # Add additional metadata
        diagnosis_metrics = {
            "total_subjects": len(predictions),
            "evaluated_subjects": len(matched_subjects),
            "matched_subjects": matched_subjects,
            "avg_confidence": 0.0,  # Calculate from predictions if needed
            **diagnosis_scores,
        }

        # Calculate average confidence from predictions
        total_confidence = 0
        for subject_id in matched_subjects:
            pred = predictions[subject_id]
            diagnosis_data = pred.get("diagnosis", {})
            total_confidence += diagnosis_data.get("confidence", 0.0)

        diagnosis_metrics["avg_confidence"] = total_confidence / len(matched_subjects)

        logger.success(
            f"Diagnosis evaluation completed: {len(matched_subjects)} subjects evaluated"
        )
        return diagnosis_metrics

    except Exception as e:
        logger.error(f"Diagnosis evaluation failed: {e}")
        return {"error": str(e)}


@beartype
def evaluate_localization_metrics(
    predictions: dict[int, dict[str, Any]], dataset: NovaDataset
) -> dict[str, Any]:
    """Evaluate localization predictions using NOVA detection evaluation metrics."""
    logger.info("🎯 Evaluating localization metrics with complete dataset...")

    # Extract predicted and reference bounding boxes
    pred_detections = []
    ref_detections = []
    matched_subjects = []

    for subject_id, pred in predictions.items():
        try:
            # Get predictions
            localization_data = pred.get("localization", {})
            pred_boxes = []
            pred_scores = []
            pred_labels = []

            if "localizations" in localization_data:
                for loc in localization_data["localizations"]:
                    if "bounding_box" in loc:
                        box = loc["bounding_box"]
                        if len(box) == 4:
                            # Keep in (x1, y1, x2, y2) format - evaluate_detection expects this
                            pred_boxes.append(list(box))
                            pred_scores.append(loc.get("confidence", 1.0))
                            pred_labels.append(1)  # Single class for abnormalities

            # Get corresponding ground truth from complete dataset
            if subject_id < len(dataset):
                sample = dataset[subject_id]
                ref_localizations = sample["ground_truth"]["localizations"]

                # Extract ground truth boxes
                ref_boxes = []
                ref_scores = []
                ref_labels = []

                for loc in ref_localizations:
                    x, y, width, height = loc["bbox"]
                    # Convert from (x, y, width, height) to (x1, y1, x2, y2)
                    ref_boxes.append([x, y, x + width, y + height])
                    ref_scores.append(1.0)  # Ground truth has perfect confidence
                    ref_labels.append(1)

                # Only include subjects that have either predictions or ground truth
                if pred_boxes or ref_boxes:
                    pred_detections.append(
                        {"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}
                    )
                    ref_detections.append(
                        {"boxes": ref_boxes, "scores": ref_scores, "labels": ref_labels}
                    )
                    matched_subjects.append(subject_id)

        except Exception as e:
            logger.warning(f"Error processing localization for subject {subject_id}: {e}")

    if not matched_subjects:
        logger.error("No valid localization pairs found for evaluation")
        return {"error": "No valid localization pairs found"}

    logger.info(f"Evaluating {len(matched_subjects)} localization pairs")

    # Evaluate using NOVA detection metrics
    try:
        detection_scores = evaluate_detection(pred_detections, ref_detections)

        # Add additional metadata
        localization_metrics = {
            "total_subjects": len(predictions),
            "evaluated_subjects": len(matched_subjects),
            "matched_subjects": matched_subjects,
            "subjects_with_predictions": sum(1 for d in pred_detections if len(d["boxes"]) > 0),
            "subjects_with_ground_truth": sum(1 for d in ref_detections if len(d["boxes"]) > 0),
            "total_predictions": sum(len(d["boxes"]) for d in pred_detections),
            "total_ground_truth": sum(len(d["boxes"]) for d in ref_detections),
            **detection_scores,
        }

        logger.success(
            f"Localization evaluation completed: {len(matched_subjects)} subjects evaluated"
        )
        return localization_metrics

    except Exception as e:
        logger.error(f"Localization evaluation failed: {e}")
        return {"error": str(e)}


@beartype
def evaluate_results(
    results_dir: str,
    output_dir: str,
    semantic_model: str = "x-ai/grok-4.1-fast:free",
) -> dict[str, Any]:
    """Main evaluation function using complete NOVA dataset."""
    logger.info(f"🔬 Evaluating results from {results_dir}")

    # Load predictions
    predictions = load_predictions(Path(results_dir))
    if not predictions:
        raise ValueError("No predictions found to evaluate")

    # Load complete NOVA dataset
    dataset = NovaDataset()
    logger.info(f"Loaded complete NOVA dataset with {len(dataset)} samples")

    # Evaluate each task
    caption_metrics = evaluate_caption_metrics(predictions, dataset)
    diagnosis_metrics = evaluate_diagnosis_metrics(predictions, dataset, semantic_model)
    localization_metrics = evaluate_localization_metrics(predictions, dataset)

    # Create comprehensive metrics
    evaluation_metrics = {
        "evaluation_info": {
            "results_directory": results_dir,
            "output_directory": output_dir,
            "total_subjects": len(predictions),
            "total_dataset_samples": len(dataset),
            "data_source": "complete_nova_dataset",
        },
        "caption_metrics": caption_metrics,
        "diagnosis_metrics": diagnosis_metrics,
        "localization_metrics": localization_metrics,
        "overall_summary": {
            "total_subjects": len(predictions),
            "evaluated_subjects": {
                "caption": caption_metrics.get("evaluated_subjects", 0),
                "diagnosis": diagnosis_metrics.get("evaluated_subjects", 0),
                "localization": localization_metrics.get("evaluated_subjects", 0),
            },
        },
    }

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed metrics
    with open(output_path / "evaluation_metrics.json", "w") as f:
        json.dump(evaluation_metrics, f, indent=2)

    # Create summary CSV
    create_summary_csv(evaluation_metrics, output_path / "summary.csv")

    logger.info("✅ Evaluation completed!")
    logger.info(f"📊 Metrics saved to: {output_path / 'evaluation_metrics.json'}")
    logger.info(f"📈 Summary CSV saved to: {output_path / 'summary.csv'}")

    return evaluation_metrics


@beartype
def create_summary_csv(metrics: dict[str, Any], output_path: Path) -> None:
    """Create CSV summary of evaluation metrics."""
    summary_data = []

    # Caption metrics
    caption = metrics.get("caption_metrics", {})
    if caption and "error" not in caption:
        summary_data.append(
            {
                "task": "caption",
                "success_rate": 1.0,  # All have captions if we got here
                "avg_confidence": caption.get("avg_confidence", 0.0),
                "subjects_total": caption.get("evaluated_subjects", 0),
            }
        )

    # Diagnosis metrics
    diagnosis = metrics.get("diagnosis_metrics", {})
    if diagnosis and "error" not in diagnosis:
        summary_data.append(
            {
                "task": "diagnosis",
                "success_rate": diagnosis.get("top1", 0.0),  # Top-1 accuracy
                "avg_confidence": diagnosis.get("avg_confidence", 0.0),
                "subjects_total": diagnosis.get("evaluated_subjects", 0),
            }
        )

    # Localization metrics
    localization = metrics.get("localization_metrics", {})
    if localization and "error" not in localization:
        summary_data.append(
            {
                "task": "localization",
                "success_rate": localization.get("acc50", 0.0),  # ACC50 is more intuitive
                "avg_confidence": 0.0,  # Localization doesn't have single confidence
                "subjects_total": localization.get("evaluated_subjects", 0),
            }
        )

    df = pd.DataFrame(summary_data)
    df.to_csv(output_path, index=False)

    # Print summary to console
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY WITH COMPLETE NOVA DATASET")
    print("=" * 70)
    print(f"Total Subjects Evaluated: {metrics['evaluation_info']['total_subjects']}")
    print(f"Total Dataset Samples: {metrics['evaluation_info']['total_dataset_samples']}")
    print("\nTask Performance (using complete dataset):")

    for data in summary_data:
        task_name = data["task"].upper()
        success_rate = data["success_rate"] * 100
        confidence = data["avg_confidence"]
        print(f"  {task_name}: {success_rate:6.1f}% accuracy, {confidence:6.3f} confidence")

    print("=" * 70)


@beartype
def evaluate_all_results(
    results_root: str,
    output_root: str,
    semantic_model: str = "x-ai/grok-4.1-fast:free",
) -> dict[str, dict[str, Any]]:
    """Evaluate all result directories under results_root."""
    results_path = Path(results_root)
    output_path = Path(output_root)

    if not results_path.exists():
        raise FileNotFoundError(f"Results root not found: {results_path}")

    # Find all subdirectories with per_subject folders
    result_dirs = [
        d for d in results_path.iterdir() if d.is_dir() and (d / "per_subject").exists()
    ]

    if not result_dirs:
        raise ValueError(f"No valid result directories found in {results_path}")

    logger.info(f"Found {len(result_dirs)} result directories to evaluate")

    all_metrics: dict[str, dict[str, Any]] = {}

    for result_dir in sorted(result_dirs):
        config_name = result_dir.name
        eval_output = output_path / config_name

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {config_name}")
        logger.info(f"{'='*60}")

        try:
            metrics = evaluate_results(str(result_dir), str(eval_output), semantic_model)
            all_metrics[config_name] = metrics
        except Exception as e:
            logger.error(f"Failed to evaluate {config_name}: {e}")
            all_metrics[config_name] = {"error": str(e)}

    # Generate comparison report
    generate_comparison_report(all_metrics, output_path)

    return all_metrics


@beartype
def generate_comparison_report(
    all_metrics: dict[str, dict[str, Any]], output_path: Path
) -> None:
    """Generate a comparison report across all configurations."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Build comparison data
    comparison_data = []

    for config_name, metrics in all_metrics.items():
        if "error" in metrics:
            continue

        row = {"configuration": config_name}

        # Caption metrics
        caption = metrics.get("caption_metrics", {})
        row["bleu"] = caption.get("bleu", 0.0)
        row["bert_f1"] = caption.get("bert_f1", 0.0)

        # Diagnosis metrics
        diagnosis = metrics.get("diagnosis_metrics", {})
        row["diagnosis_top1"] = diagnosis.get("top1", 0.0)
        row["diagnosis_top5"] = diagnosis.get("top5", 0.0)

        # Localization metrics
        localization = metrics.get("localization_metrics", {})
        row["map30"] = localization.get("map30", 0.0)
        row["map50"] = localization.get("map50", 0.0)
        row["acc50"] = localization.get("acc50", 0.0)

        comparison_data.append(row)

    if not comparison_data:
        logger.warning("No valid metrics to compare")
        return

    # Create comparison DataFrame
    df = pd.DataFrame(comparison_data)
    df = df.sort_values("configuration")

    # Save CSV
    csv_path = output_path / "comparison.csv"
    df.to_csv(csv_path, index=False)

    # Save JSON
    json_path = output_path / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Print comparison table
    print("\n" + "=" * 100)
    print("📊 COMPARISON ACROSS ALL CONFIGURATIONS")
    print("=" * 100)
    print(
        f"{'Configuration':<45} {'Top-1':>8} {'Top-5':>8} {'ACC50':>8} "
        f"{'mAP@50':>8} {'BLEU':>8}"
    )
    print("-" * 100)

    for _, row in df.iterrows():
        print(
            f"{row['configuration']:<45} "
            f"{row['diagnosis_top1']*100:>7.1f}% "
            f"{row['diagnosis_top5']*100:>7.1f}% "
            f"{row['acc50']*100:>7.1f}% "
            f"{row['map50']*100:>7.1f}% "
            f"{row['bleu']*100:>7.1f}%"
        )

    print("=" * 100)
    logger.info(f"\n📁 Comparison saved to: {csv_path}")


def main() -> None:
    """Main evaluation script entry point."""
    parser = ArgumentParser(description="Evaluate NOVA predictions against complete dataset")
    parser.add_argument(
        "--results-dir",
        help="Directory containing prediction results (single config)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Evaluate all subdirectories under results-dir",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for evaluation metrics",
    )
    parser.add_argument(
        "--semantic-model",
        default="x-ai/grok-4.1-fast:free",
        help="Model for diagnosis semantic matching (default: x-ai/grok-4.1-fast:free)",
    )

    args = parser.parse_args()

    # Default to ./results if no results-dir specified with --batch
    results_dir = args.results_dir or ("./results" if args.batch else None)

    if not results_dir:
        parser.error("--results-dir is required unless using --batch")

    try:
        if args.batch:
            evaluate_all_results(results_dir, args.output, args.semantic_model)
        else:
            evaluate_results(results_dir, args.output, args.semantic_model)
        logger.success("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
