from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw


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
