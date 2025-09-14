"""Shared I/O + visualisation utilities used by multiple pipelines.

This module centralises boilerplate that was previously duplicated across
`nova_retrieval_vlm.cli.process_batch_*` functions:

• image-folder preparation
• ensure_evaluation_keys / save_prediction / save_reference
• bounding-box visualisation
• per-image evaluation

Consolidating them guarantees consistent behaviour across all operational
modes and reduces maintenance overhead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from loguru import logger

if TYPE_CHECKING:
    from matplotlib.lines import Line2D


@dataclass
class BatchContext:  # noqa: D101 – simple data container
    idx: int
    folder: Path
    img_path: Path  # Path to the persisted PNG
    width: int
    height: int


# ---------------------------------------------------------------------------
# Core helpers (moved from cli.py verbatim where relevant)
# ---------------------------------------------------------------------------


def convert_localization_schema(result: dict) -> None:
    """Convert new localization schema to legacy format for backward compatibility."""
    if "localizations" in result and "boxes" not in result:
        localizations = result["localizations"]
        boxes = []
        labels = []
        scores = []

        for loc in localizations:
            if "bounding_box" in loc:
                boxes.append(loc["bounding_box"])
                labels.append("anomaly")  # Standard label for compatibility
                scores.append(loc.get("confidence", 1.0))

        result["boxes"] = boxes
        result["labels"] = labels
        result["scores"] = scores


def normalize_localization_result(result: dict) -> None:  # noqa: D401
    """Add *boxes* / *labels* / *scores* keys if missing (in-place)."""

    # First convert new schema to legacy if needed
    convert_localization_schema(result)

    if "boxes" not in result or result["boxes"] is None:
        result["boxes"] = []
    if "labels" not in result or result["labels"] is None:
        result["labels"] = []
    if "scores" not in result or result["scores"] is None:
        result["scores"] = []

    # Standardise labels & scores length if caller only provided boxes
    boxes_len = len(result["boxes"])
    if len(result["labels"]) != boxes_len:
        result["labels"] = ["anomaly"] * boxes_len
    if len(result["scores"]) != boxes_len:
        result["scores"] = [1.0] * boxes_len


def save_prediction(img_folder: Path, result: dict) -> None:  # noqa: D401
    pred_file = img_folder / "pred.jsonl"
    with pred_file.open("w") as fw:
        fw.write(json.dumps(result) + "\n")


def save_reference(img_folder: Path, batch_idx: int, hf_ds) -> None:  # noqa: D401
    ref_file = img_folder / "ref.jsonl"
    rec = hf_ds[batch_idx]
    bg = rec.get("bbox_gold", {})
    boxes = [
        [x, y, x + w, y + h]
        for x, y, w, h in zip(
            bg.get("x", []),
            bg.get("y", []),
            bg.get("width", []),
            bg.get("height", []),
            strict=False,
        )
    ]
    labels = ["anomaly"] * len(boxes)
    scores = [1.0] * len(boxes)
    caption = rec.get("caption", "")
    diagnosis = rec.get("final_diagnosis") or rec.get("diagnosis", "")
    ref_data = {
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
        "caption": caption,
        "diagnosis": diagnosis,
        "ground_truth_image_idx": batch_idx,
    }
    with ref_file.open("w") as fr:
        fr.write(json.dumps(ref_data) + "\n")


def _iter_boxes_generic(raw_boxes: list[Any]):
    """Yield boxes in [x1,y1,x2,y2] regardless of input encoding."""

    for b in raw_boxes:
        if isinstance(b, list | tuple) and len(b) == 4:
            yield b
        elif isinstance(b, dict):
            if all(k in b for k in ("x1", "y1", "x2", "y2")):
                yield [b["x1"], b["y1"], b["x2"], b["y2"]]
            elif all(k in b for k in ("x", "y", "width", "height")):
                yield [b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]]


def draw_ground_truth_vs_predicted_boxes(
    img_path: Path,
    gt_boxes: list[Any],
    pred_boxes: list[Any],
    out_path: Path,
) -> None:
    """Create *out_path* PNG with GT (green) and pred (red) boxes."""

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib import patheffects as pe
    from matplotlib.lines import Line2D
    from PIL import Image

    matplotlib.use("Agg")  # headless backend

    img = Image.open(img_path).convert("L")
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img, cmap="gray")

    for x1, y1, x2, y2 in _iter_boxes_generic(gt_boxes):
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.5,
                edgecolor="lime",
                facecolor="none",
            )
        )

    for x1, y1, x2, y2 in _iter_boxes_generic(pred_boxes):
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.2,
                edgecolor="red",
                linestyle="--",
                facecolor="none",
            )
        )

    legend_elems: list[Line2D] = []
    if gt_boxes:
        legend_elems.append(Line2D([0], [0], color="lime", lw=2, label="Ground Truth"))
    if pred_boxes:
        legend_elems.append(Line2D([0], [0], color="red", lw=2, linestyle="--", label="Prediction"))
    if legend_elems:
        leg = ax.legend(handles=legend_elems, loc="upper right", fontsize="x-small", frameon=False)
        for txt in leg.get_texts():
            txt.set_color("yellow")
            txt.set_path_effects([pe.Stroke(linewidth=1.0, foreground="black"), pe.Normal()])

    ax.axis("off")
    fig.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def compute_evaluation_metrics(img_folder: Path, task: str) -> None:  # noqa: D401
    """Run evaluation for this image and write metrics.json."""

    from nova_retrieval_vlm.evaluation import evaluate  # local import

    pred_file = img_folder / "pred.jsonl"
    ref_file = img_folder / "ref.jsonl"

    if not pred_file.exists() or not ref_file.exists():
        raise FileNotFoundError(f"Missing prediction or reference file in {img_folder}")

    single_metrics = evaluate(str(pred_file), str(ref_file), task=task)
    logger.info("Evaluation metrics for %s: %s", img_folder.name, single_metrics)

    with (img_folder / "metrics.json").open("w") as f:
        json.dump(single_metrics, f, indent=2)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def postprocess_batch_result(
    ctx: BatchContext,
    result: dict,
    task: str,
    hf_ds,
    preds: list,
):
    """One-liner to perform all common tail-steps for a batch."""

    normalize_localization_result(result)
    save_prediction(ctx.folder, result)
    save_reference(ctx.folder, ctx.idx, hf_ds)

    # Draw visualisation and run evaluation
    gt_bg = hf_ds[ctx.idx].get("bbox_gold", {})
    gt_boxes = [
        [x, y, x + w, y + h]
        for x, y, w, h in zip(
            gt_bg.get("x", []),
            gt_bg.get("y", []),
            gt_bg.get("width", []),
            gt_bg.get("height", []),
            strict=False,
        )
    ]
    draw_ground_truth_vs_predicted_boxes(
        ctx.img_path, gt_boxes, result.get("boxes", []), ctx.folder / "bboxes.png"
    )
    compute_evaluation_metrics(ctx.folder, task)

    preds.append(result)
