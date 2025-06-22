from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import OmegaConf

from nova_retrieval_vlm.config import Config
from nova_retrieval_vlm.data.nova_dataset import get_dataloader
from nova_retrieval_vlm.evaluation import evaluate
from nova_retrieval_vlm.guidelines.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.prompts.prompt_loader import load_prompt
from nova_retrieval_vlm.visual_reasoning.image_ops import (
    adjust_contrast,
    apply_intensity_threshold,
    crop_image,
    zoom_image,
)

# ---------------------------------------------------------------------------
# Optional .env loading
# ---------------------------------------------------------------------------
#
# `python-dotenv` is convenient for local development but not a hard runtime
# requirement.  We therefore attempt to import it **lazily** and fall back to a
# no-op when the package is absent so that the CLI still works inside minimal
# Docker images or freshly created virtual-envs.

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
    logger.debug("Loaded environment variables from .env file (python-dotenv).")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    logger.warning(
        "python-dotenv not installed - skipping automatic '.env' loading. "
        "Environment variables must be set explicitly."
    )

# Register structured config
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base="1.1", config_path=None, config_name="config")
def main(cfg: Config) -> None:
    """
    Main entry point for running experiments.

    Usage examples:
      python -m nova_retrieval_vlm.cli experiment=baseline model=qwen-vl-chat batch_size=4
      python -m nova_retrieval_vlm.cli experiment=hybrid model=internvlm-chat retrieval.top_k=8
    """
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    data_dir = to_absolute_path(cfg.paths.data_dir)
    output_dir = to_absolute_path(cfg.paths.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Handle free-form prompt-only task
    if cfg.task == "prompt":
        adapter = setup_adapter(cfg)
        text, log = asyncio.run(adapter.generate_text(cfg.prompt_text))
        logger.info(f"Generation cost: {log}")
        print(text)
        return

    # Handle visualization task and exit early
    if cfg.task == "visualize":
        from nova_retrieval_vlm.visualization.sample_utils import visualize_samples

        visualize_samples(
            num_samples=cfg.visualization.num_samples,
            out_dir=output_dir,
            cache_dir=data_dir,
            trust_remote_code=cfg.visualization.trust_remote_code,
            overlay=cfg.visualization.overlay,
        )
        return

    # Load data (always test split zero-shot)
    dl = get_dataloader(
        batch_size=cfg.batch_size,
        data_dir=data_dir,
    )
    task = cfg.task  # 'baseline','retrieval','localization','caption','diagnosis'

    # Setup retriever if retrieval augmentation is enabled
    retriever = setup_retriever(cfg)

    # Setup adapter with fallback support
    adapter = setup_adapter(cfg)

    preds: list[dict] = []
    # Iterate over dataset
    # ---- Set iteration limit for testing or full dataset ----
    max_iterations = cfg.max_iterations if cfg.max_iterations > 0 else float("inf")
    current_iteration = 0

    # Create a main run directory with timestamp
    main_run_dir = create_run_directory(output_dir)

    # ---------------------------------------------------------------------
    # Reference dataset (needed for creating refs.jsonl)
    # ---------------------------------------------------------------------

    local_path = Path(to_absolute_path(cfg.paths.data_dir)) / "nova_test"
    logger.info(f"Looking for pre-processed arrow dataset under: {local_path}")

    if local_path.exists():
        # Preferred fast-path - load the arrow file we previously generated
        hf_ds = load_dataset("arrow", data_files=str(local_path / "data-00000-of-00001.arrow"))[
            "train"
        ]
        logger.info("Loaded cached arrow dataset with %d samples", len(hf_ds))
    else:
        # Fallback: use the HuggingFace dataset that was already downloaded by
        # NovaDataset (exposed via dl.dataset.dataset).
        logger.warning("Cached arrow dataset not found - falling back to in-memory HF dataset.")
        # dl.dataset is an instance of NovaDataset; we expose the underlying HF
        # dataset via the public attribute `dataset`.
        try:
            hf_ds = dl.dataset.dataset  # type: ignore[attr-defined]
            logger.info("Using HF dataset from NovaDataset with %d samples", len(hf_ds))
        except Exception as exc:
            logger.error("Unable to access HF dataset from DataLoader: %s", exc)
            raise FileNotFoundError(
                "Could not locate a reference NOVA dataset (neither cached arrow nor in-memory)."
            ) from exc

    for batch_idx, batch in enumerate(dl):
        # Check if we've reached max iterations
        if current_iteration >= max_iterations:
            logger.info(f"Reached MAX_ITERATIONS ({max_iterations}), breaking loop.")
            break

        # Process each batch according to the chosen *approach*
        if cfg.approach == "baseline":
            process_batch(
                batch_idx=batch_idx,
                batch=batch,
                main_run_dir=main_run_dir,
                task=task,
                cfg=cfg,
                adapter=adapter,
                retriever=retriever,
                preds=preds,
                hf_ds=hf_ds,
            )
        elif cfg.approach == "multiturn":
            process_batch_multiturn(
                batch_idx=batch_idx,
                batch=batch,
                main_run_dir=main_run_dir,
                task=task,
                cfg=cfg,
                adapter=adapter,
                retriever=retriever,
                preds=preds,
                hf_ds=hf_ds,
            )
        elif cfg.approach == "visual_multiturn":
            process_batch_visual_multiturn(
                batch_idx=batch_idx,
                batch=batch,
                main_run_dir=main_run_dir,
                task=task,
                cfg=cfg,
                adapter=adapter,
                retriever=retriever,
                preds=preds,
                hf_ds=hf_ds,
            )
        else:
            raise ValueError(f"Unknown approach: {cfg.approach}")

        # Add a delay to help with rate limiting
        time.sleep(cfg.request_delay)

        # Increment counter
        current_iteration += 1

    logger.info(f"Finished processing {len(preds)} predictions. Performing overall evaluation...")

    # Save predictions
    ts = int(time.time())
    run_dir = Path(output_dir) / str(ts)
    logger.info(f"Creating run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    preds_file = run_dir / "preds.jsonl"
    logger.info(f"Saving predictions to: {preds_file}")
    with open(preds_file, "w") as fw:
        for pred in preds:
            # Standardize for evaluation
            if isinstance(pred, dict):
                boxes_len = len(pred.get("boxes", []))
                if "labels" not in pred:
                    pred["labels"] = ["anomaly"] * boxes_len
                if "scores" not in pred:
                    pred["scores"] = [1.0] * boxes_len
            fw.write(json.dumps(pred) + "\n")

    # Create references file
    refs_file = run_dir / "refs.jsonl"
    logger.info(f"Saving references to: {refs_file}")
    with open(refs_file, "w") as fr:
        for i, rec in enumerate(hf_ds):
            if i >= len(preds):
                break
            bg = rec.get("bbox_gold", {})
            boxes = [
                [x, y, x + w, y + h]
                for x, y, w, h in zip(
                    bg.get("x", []), bg.get("y", []), bg.get("width", []), bg.get("height", [])
                )
            ]
            labels = ["anomaly"] * len(boxes)
            scores = [1.0] * len(boxes)
            caption = rec.get("caption", "")
            diagnosis = rec.get("final_diagnosis") or rec.get("diagnosis", "")
            fr.write(
                json.dumps(
                    {
                        "boxes": boxes,
                        "labels": labels,
                        "scores": scores,
                        "caption": caption,
                        "diagnosis": diagnosis,
                        "ground_truth_image_idx": i,
                    }
                )
                + "\n"
            )

    # Overall evaluation - propagate ImportError so that missing metric
    # dependencies cause an explicit crash (fail-fast policy).
    metrics = evaluate(str(preds_file), str(refs_file), task=cfg.task)
    logger.info(f"Overall evaluation metrics: {metrics}")

    # Return summary of the run
    return {"run_dir": str(run_dir), "metrics": metrics}


def setup_adapter(cfg: Config):
    """Set up the model adapter without fallback support."""
    logger.info(f"Setting up adapter for model: {cfg.model.name}")
    return OpenAIAdapter(
        model_name=cfg.model.name, max_retries=cfg.model.max_retries, timeout=cfg.model.timeout
    )


def setup_retriever(cfg: Config):
    """Set up the retriever based on configuration."""
    if not cfg.use_retrieval:
        return None

    bm25_idx = Path(to_absolute_path(cfg.paths.index_dir)) / "bm25"
    faiss_idx = Path(to_absolute_path(cfg.paths.index_dir)) / "faiss"

    try:
        if cfg.retrieval.type == "bm25":
            logger.info(f"Setting up BM25Retriever from {bm25_idx}")
            return BM25Retriever(str(bm25_idx))
        elif cfg.retrieval.type == "dense":
            logger.info(f"Setting up DenseRetriever from {faiss_idx}")
            return DenseRetriever(str(faiss_idx))
        else:
            logger.info(f"Setting up HybridRetriever from {bm25_idx} and {faiss_idx}")
            return HybridRetriever(
                BM25Retriever(str(bm25_idx)),
                DenseRetriever(str(faiss_idx)),
                alpha=cfg.retrieval.hybrid_ratio,
            )
    except Exception as e:
        logger.error(f"Failed to setup retriever (type: {cfg.retrieval.type}): {e}")
        logger.error("This will cause retrieval to fail silently. Consider:")
        logger.error("1. Checking if indexes exist in the specified directory")
        logger.error("2. Running the index building script first")
        logger.error("3. Checking Haystack version compatibility")
        raise


def create_run_directory(output_dir: str) -> Path:
    """Create a timestamped run directory."""
    ts = int(time.time())
    main_run_dir = Path(output_dir) / str(ts)
    logger.info(f"Creating main run directory: {main_run_dir}")
    main_run_dir.mkdir(parents=True, exist_ok=True)
    return main_run_dir


def process_batch(
    batch_idx: int,
    batch: dict,
    main_run_dir: Path,
    task: str,
    cfg: Config,
    adapter,
    retriever,
    preds: list,
    hf_ds,
):
    """Process a single batch from the dataloader."""
    # ------------------------------------------------------------------
    # Persist the image at its ORIGINAL resolution - critical for proper
    # bounding-box visualisation.  We therefore reload the image directly
    # from the HuggingFace dataset (which retains the full-size image) and
    # bypass any resize transforms that were applied for model ingestion.
    # ------------------------------------------------------------------

    from PIL import Image  # local import to avoid top-level PIL dependency

    hf_record = hf_ds[batch_idx]

    if "image" in hf_record and hf_record["image"] is not None:
        pil = hf_record["image"]
        if not isinstance(pil, Image.Image):
            pil = Image.fromarray(pil)
        pil = pil.convert("L")  # ensure grayscale
    else:
        # Fallback: load from stored path in dataset record
        img_path_str = hf_record.get("image_path")
        if not img_path_str:
            raise ValueError(f"No image or image_path field for record {batch_idx}")
        pil = Image.open(img_path_str).convert("L")

    # Create a folder for this specific image_id or batch index
    img_folder = main_run_dir / f"image_{batch_idx}"
    img_folder.mkdir(parents=True, exist_ok=True)
    img_path = img_folder / "image.png"
    pil.save(img_path)

    # Baseline run: we do **not** augment with retrieval passages.
    passages: list[str] = []

    # Select baseline prompt template (no retrieval variant)
    template_name = f"baseline/{task}.jinja"

    # Prepare metadata for the prompt, always including image_id and dims
    metadata_for_prompt = {
        "image_id": batch_idx,
        "width": pil.width,
        "height": pil.height,
    }
    if task == "diagnosis":
        metadata_for_prompt["clinical_history"] = batch["metadata"][0].get("clinical_history", "")

    # Load prompt
    prompt = load_prompt(template_name, img_path, passages, metadata_for_prompt)
    # Log the prompt to a file for later inspection
    with open(img_folder / "prompt.txt", "w") as f:
        f.write(prompt)
    logger.debug(f"Rendered prompt for image {batch_idx}")

    # Generate response
    text, log = asyncio.run(adapter.generate(img_path, passages, system_prompt=prompt))
    logger.info(f"Generation cost for image {batch_idx}: {log}")

    # Save raw output for debugging
    with open(img_folder / "raw_output.txt", "w") as f:
        f.write(text)

    # -------------------------------------------------------------
    # Parse JSON with robustness to badly-formatted model output
    # -------------------------------------------------------------

    def _safe_json_loads(payload: str) -> dict:
        """Attempt to parse *payload* as JSON.

        The VLM sometimes emits stray characters (e.g. leading "Answer:" lines)
        or wrongly formatted Markdown.  We therefore try progressively more
        forgiving strategies before giving up:

        1. Direct `json.loads`.
        2. Find the first '{{' and last '}}' and try to load that substring.
        3. Return a minimal fallback dict containing the raw text.
        """
        import re

        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            # Strategy 2: extract first/last curly braces block
            m = re.search(r"\{.*\}", payload, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        # Final fallback - return raw text so that downstream still works.
        logger.warning("Unable to parse model output as JSON for image %d", batch_idx)
        return {"raw": payload, "boxes": [], "labels": [], "scores": []}

    result = _safe_json_loads(text)

    # Add metadata
    result["image_path"] = str(img_path)
    result["ground_truth_image_idx"] = batch_idx

    # Ensure required keys are present
    ensure_evaluation_keys(result)

    logger.info(f"Successfully processed result for image {batch_idx}")

    # Save individual prediction for this image
    save_prediction(img_folder, result)

    # Generate and save reference
    save_reference(img_folder, batch_idx, hf_ds)

    # ---------------------------------------------------------------------
    # Visualisation - draw GT & predicted bounding-boxes
    # ---------------------------------------------------------------------

    def _draw_boxes(img_path: Path, gt: list[Any], pred: list[Any], out_path: Path):
        """Draw *gt* (green) and *pred* (red) boxes on *img_path* and save.

        The helper is now tolerant to various box formats:

        1. [x1, y1, x2, y2] - preferred.
        2. Dicts with keys (x1,y1,x2,y2) or (x,y,width,height).
        Invalid or incomplete entries are silently skipped.
        """
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from matplotlib import patheffects as pe
        from PIL import Image

        img = Image.open(img_path).convert("L")
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(img, cmap="gray")

        def _iter_boxes(raw_boxes):
            for b in raw_boxes:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    yield b
                elif isinstance(b, dict):
                    if all(k in b for k in ("x1", "y1", "x2", "y2")):
                        yield [b["x1"], b["y1"], b["x2"], b["y2"]]
                    elif all(k in b for k in ("x", "y", "width", "height")):
                        yield [b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]]

        for box in _iter_boxes(gt):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), w, h, linewidth=1.5, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

        for box in _iter_boxes(pred):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle(
                (x1, y1), w, h, linewidth=1.2, edgecolor="red", facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)

        ax.axis("off")

        # ------------------------------------------------------------------
        # Add legend (only for categories that are present)
        # ------------------------------------------------------------------
        from matplotlib.lines import Line2D  # imported here to keep the

        # function self-contained even when matplotlib is not globally
        # available at import time.

        legend_elements = []
        if gt:
            legend_elements.append(Line2D([0], [0], color="lime", lw=2, label="Ground Truth"))
        if pred:
            legend_elements.append(
                Line2D([0], [0], color="red", lw=2, linestyle="--", label="Prediction")
            )

        if legend_elements:
            leg = ax.legend(
                handles=legend_elements, loc="upper right", fontsize="x-small", frameon=False
            )
            # Improve readability - bright text with thin black outline
            for text in leg.get_texts():
                text.set_color("yellow")
                text.set_path_effects(
                    [
                        pe.Stroke(linewidth=1.0, foreground="black"),
                        pe.Normal(),
                    ]
                )

        fig.tight_layout(pad=0)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    gt_bg = hf_ds[batch_idx].get("bbox_gold", {})
    gt_boxes = [
        [x, y, x + w, y + h]
        for x, y, w, h in zip(
            gt_bg.get("x", []), gt_bg.get("y", []), gt_bg.get("width", []), gt_bg.get("height", [])
        )
    ]
    viz_path = img_folder / "bboxes.png"
    _draw_boxes(img_path, gt_boxes, result.get("boxes", []), viz_path)

    # Evaluate this individual prediction
    evaluate_prediction(img_folder, task)

    # Add to combined predictions list
    preds.append(result)


def ensure_evaluation_keys(result: dict):
    """Ensure all required keys are present in the result dict."""
    if "boxes" not in result:
        result["boxes"] = []
    if "labels" not in result:
        result["labels"] = []
    if "scores" not in result:
        # Default scores to 1.0 for all predicted boxes if not provided
        result["scores"] = [1.0] * len(result["boxes"])

    # Standardize labels to 'anomaly' for all boxes
    boxes_len = len(result.get("boxes", []))
    result["labels"] = ["anomaly"] * boxes_len
    result["scores"] = [1.0] * boxes_len


def save_prediction(img_folder: Path, result: dict):
    """Save prediction to file."""
    pred_file = img_folder / "pred.jsonl"
    with open(pred_file, "w") as fw:
        fw.write(json.dumps(result) + "\n")


def save_reference(img_folder: Path, batch_idx: int, hf_ds):
    """Save reference to file."""
    ref_file = img_folder / "ref.jsonl"
    rec = hf_ds[batch_idx]
    bg = rec.get("bbox_gold", {})
    boxes = [
        [x, y, x + w, y + h]
        for x, y, w, h in zip(
            bg.get("x", []), bg.get("y", []), bg.get("width", []), bg.get("height", [])
        )
    ]
    # Standardize reference labels to 'anomaly' and scores to 1.0
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
    with open(ref_file, "w") as fr:
        fr.write(json.dumps(ref_data) + "\n")


def evaluate_prediction(img_folder: Path, task: str):
    """Evaluate single prediction against reference."""
    pred_file = img_folder / "pred.jsonl"
    ref_file = img_folder / "ref.jsonl"

    if not pred_file.exists() or not ref_file.exists():
        raise FileNotFoundError(f"Missing prediction or reference file in {img_folder}")

    single_metrics = evaluate(str(pred_file), str(ref_file), task=task)
    logger.info("Evaluation metrics for %s: %s", img_folder.name, single_metrics)

    # Save individual metrics
    with open(img_folder / "metrics.json", "w") as f:
        json.dump(single_metrics, f, indent=2)


# ======================================================================
# Experimental multi-turn controller - *stub* version.
# ----------------------------------------------------------------------
# For now it simply proxies to the baseline `process_batch`.  We keep a
# separate entry-point so that future work can implement the full dialogue
# loop (clarifying questions, self-reflection, guideline retrieval, etc.)
# without touching the main training/eval loop.
# ======================================================================


def process_batch_multiturn(
    batch_idx: int,
    batch: dict,
    main_run_dir: Path,
    task: str,
    cfg: Config,
    adapter,
    retriever,
    preds: list,
    hf_ds,
):
    """Multi-turn variant with iterative retrieval support.

    The function implements a clinical workflow with:
    1. Initial observations and differential diagnosis
    2. Guideline retrieval and analysis
    3. Optional additional retrieval requests (up to 3 rounds)
    4. Final task-specific output (localization, caption, or diagnosis)
    """

    logger.debug("[multiturn] Starting iterative workflow for image %d", batch_idx)

    # ------------------------------------------------------------------
    # 1. Prepare common resources (image path, retrieval passages, etc.)
    # ------------------------------------------------------------------

    from PIL import Image  # local import

    hf_record = hf_ds[batch_idx]

    if "image" in hf_record and hf_record["image"] is not None:
        pil = hf_record["image"]
        if not isinstance(pil, Image.Image):
            pil = Image.fromarray(pil)
        pil = pil.convert("L")
    else:
        img_path_str = hf_record.get("image_path")
        if not img_path_str:
            raise ValueError(f"No image or image_path field for record {batch_idx}")
        pil = Image.open(img_path_str).convert("L")

    img_folder = main_run_dir / f"image_{batch_idx}"
    img_folder.mkdir(parents=True, exist_ok=True)
    img_path = img_folder / "image.png"
    pil.save(img_path)

    width, height = pil.width, pil.height

    # Metadata used in templates
    metadata_common = {
        "image_id": batch_idx,
        "width": width,
        "height": height,
    }

    # Add clinical history for diagnosis task
    if task == "diagnosis":
        metadata_common["clinical_history"] = batch["metadata"][0].get("clinical_history", "")

    # ------------------------------------------------------------------
    # 2. TURN-1  (initial observations + provisional differential)
    # ------------------------------------------------------------------

    prompt_turn1 = load_prompt(
        "multiturn/step1.jinja",
        img_path=img_path,
        passages=[],
        metadata=metadata_common,
    )

    logger.debug("[multiturn] Rendering TURN-1 prompt for image %d", batch_idx)

    text1, log1 = asyncio.run(adapter.generate(img_path, passages=[], system_prompt=prompt_turn1))

    with open(img_folder / "turn1_raw.txt", "w") as f:
        f.write(text1)

    # Parse JSON safely
    def _safe_json_loads(payload: str):
        import json as _json
        import re

        try:
            return _json.loads(payload)
        except _json.JSONDecodeError:
            m = re.search(r"\{.*\}", payload, re.DOTALL)
            if m:
                try:
                    return _json.loads(m.group(0))
                except _json.JSONDecodeError:
                    pass
        logger.warning("[multiturn] Could not parse JSON for image %d", batch_idx)
        return {}

    turn1_data = _safe_json_loads(text1)
    diff_list = turn1_data.get("differential", []) if isinstance(turn1_data, dict) else []

    # Extract summary for iterative retrieval
    step1_summary = turn1_data.get("summary", "") if isinstance(turn1_data, dict) else ""

    # ------------------------------------------------------------------
    # 3. Initial guideline retrieval for differential
    # ------------------------------------------------------------------

    passages: list[str] = []
    retrieval_debug_info = {
        "query": None,
        "differential_from_turn1": diff_list,
        "retrieval_success": False,
        "num_passages_retrieved": 0,
        "error": None,
        "passages": [],
    }

    if diff_list and retriever:
        query = ", ".join(diff_list)
        retrieval_debug_info["query"] = query
        try:
            passages = retriever(query, k=cfg.retrieval.top_k)
            retrieval_debug_info["retrieval_success"] = True
            retrieval_debug_info["num_passages_retrieved"] = len(passages)
            retrieval_debug_info["passages"] = passages
        except Exception as exc:
            logger.warning("[multiturn] Initial retrieval failed: %s", exc)
            retrieval_debug_info["error"] = str(exc)

    # Save initial retrieval debug info
    with open(img_folder / "retrieval_debug.txt", "w") as f:
        f.write("=== INITIAL RETRIEVAL DEBUG INFO ===\n\n")
        f.write(f"Image ID: {batch_idx}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Timestamp: {__import__('datetime').datetime.now()}\n\n")

        f.write("=== TURN 1 DIFFERENTIAL ===\n")
        f.write(f"Raw differential: {diff_list}\n\n")

        f.write("=== RETRIEVAL QUERY ===\n")
        f.write(f"Query: {retrieval_debug_info['query']}\n")
        f.write(f"Retrieval success: {retrieval_debug_info['retrieval_success']}\n")
        f.write(f"Number of passages: {retrieval_debug_info['num_passages_retrieved']}\n")
        if retrieval_debug_info["error"]:
            f.write(f"Error: {retrieval_debug_info['error']}\n")
        f.write("\n")

        f.write("=== RETRIEVED PASSAGES ===\n")
        if passages:
            for i, passage in enumerate(passages, 1):
                f.write(f"--- Passage {i} ---\n")
                f.write(passage)
                f.write("\n\n")
        else:
            f.write("No passages retrieved.\n")

    # Save passages in simple format
    if passages:
        with open(img_folder / "passages.txt", "w") as f:
            f.write("\n\n".join(passages))

    # ------------------------------------------------------------------
    # 4. TURN-2  (summary reasoning with guidelines)
    # ------------------------------------------------------------------

    prompt_turn2 = load_prompt(
        "multiturn/step2.jinja",
        img_path=img_path,
        passages=passages,
        metadata=metadata_common,
    )

    text2, log2 = asyncio.run(
        adapter.generate(img_path, passages=passages, system_prompt=prompt_turn2)
    )

    with open(img_folder / "turn2_raw.txt", "w") as f:
        f.write(text2)

    # Extract summary for iterative retrieval
    step2_summary = text2  # Use full text as summary for now

    # ------------------------------------------------------------------
    # 5. Optional Step 2b (clinical history integration)
    # ------------------------------------------------------------------

    clinical_history_integration = ""
    if task == "diagnosis" and metadata_common.get("clinical_history"):
        prompt_step2b = load_prompt(
            "multiturn/step2b_clinical_history.jinja",
            img_path=img_path,
            passages=passages,
            metadata=metadata_common,
        )

        text2b, log2b = asyncio.run(
            adapter.generate(img_path, passages=passages, system_prompt=prompt_step2b)
        )

        with open(img_folder / "turn2b_raw.txt", "w") as f:
            f.write(text2b)

        clinical_history_integration = text2b

    # ------------------------------------------------------------------
    # 6. Iterative retrieval loop (up to 3 additional rounds)
    # ------------------------------------------------------------------

    all_passages = passages.copy()
    retrieval_history = []
    request_count = 0
    max_additional_requests = 3

    while request_count < max_additional_requests and retriever:
        # Prepare metadata for retrieval request
        retrieval_metadata = {
            **metadata_common,
            "step1_summary": step1_summary,
            "step2_summary": step2_summary,
            "clinical_history_integration": clinical_history_integration,
            "request_count": request_count,
            "additional_passages": all_passages,
            "retrieval_history": retrieval_history,
        }

        # Step 2c: Ask model if it needs additional retrieval
        prompt_step2c = load_prompt(
            "multiturn/step2c_retrieval_request.jinja",
            img_path=img_path,
            passages=all_passages,
            metadata=retrieval_metadata,
        )

        text2c, log2c = asyncio.run(
            adapter.generate(img_path, passages=all_passages, system_prompt=prompt_step2c)
        )

        with open(img_folder / f"turn2c_request_{request_count + 1}_raw.txt", "w") as f:
            f.write(text2c)

        # Parse retrieval request
        retrieval_request = _safe_json_loads(text2c)

        need_additional = retrieval_request.get("need_additional_retrieval", False)
        proceed_to_final = retrieval_request.get("proceed_to_final_steps", True)

        if not need_additional or proceed_to_final:
            logger.debug(
                "[multiturn] Model satisfied with current information, proceeding to final steps"
            )
            break

        # Process additional retrieval requests
        retrieval_requests = retrieval_request.get("retrieval_requests", [])
        if not retrieval_requests:
            logger.debug(
                "[multiturn] No specific retrieval requests provided, proceeding to final steps"
            )
            break

        # Execute additional retrievals
        additional_passages_round = []
        for req in retrieval_requests:
            search_terms = req.get("search_terms", [])
            if search_terms:
                query = ", ".join(search_terms)
                try:
                    new_passages = retriever(query, k=cfg.retrieval.top_k)
                    additional_passages_round.extend(new_passages)

                    # Log this retrieval
                    retrieval_history.append(
                        {
                            "request_number": request_count + 1,
                            "search_terms": search_terms,
                            "query": query,
                            "justification": req.get("justification", ""),
                            "expected_benefit": req.get("expected_benefit", ""),
                            "passages_retrieved": len(new_passages),
                            "passages": new_passages,
                        }
                    )

                    logger.debug(
                        "[multiturn] Additional retrieval %d: %s -> %d passages",
                        request_count + 1,
                        query,
                        len(new_passages),
                    )

                except Exception as exc:
                    logger.warning("[multiturn] Additional retrieval failed: %s", exc)

        # Add new passages to collection
        if additional_passages_round:
            all_passages.extend(additional_passages_round)
            # Remove duplicates while preserving order
            seen = set()
            unique_passages = []
            for passage in all_passages:
                if passage not in seen:
                    seen.add(passage)
                    unique_passages.append(passage)
            all_passages = unique_passages

            # Save updated passages
            with open(img_folder / f"passages_after_request_{request_count + 1}.txt", "w") as f:
                f.write("\n\n".join(all_passages))

        request_count += 1

    # Save final retrieval history
    with open(img_folder / "retrieval_history.json", "w") as f:
        import json

        json.dump(
            {
                "total_requests": request_count,
                "total_passages": len(all_passages),
                "retrieval_history": retrieval_history,
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 7. TURN-3  (task-specific output with all accumulated passages)
    # ------------------------------------------------------------------

    step3_template_map = {
        "localization": "multiturn/localization_step3.jinja",
        "caption": "multiturn/caption_step3.jinja",
        "diagnosis": "multiturn/diagnosis_step3.jinja",
    }

    tmpl_name3 = step3_template_map.get(task)
    if tmpl_name3 is None:
        raise ValueError(f"Unsupported task for multiturn approach: {task}")

    # Prepare final metadata with all accumulated information
    final_metadata = {
        **metadata_common,
        "step1_summary": step1_summary,
        "step2_summary": step2_summary,
        "clinical_history_integration": clinical_history_integration,
        "request_count": request_count,
        "additional_passages": all_passages,
        "retrieval_history": retrieval_history,
    }

    prompt_turn3 = load_prompt(
        tmpl_name3,
        img_path=img_path,
        passages=all_passages,  # Use all accumulated passages
        metadata=final_metadata,
    )

    text3, log3 = asyncio.run(
        adapter.generate(img_path, passages=all_passages, system_prompt=prompt_turn3)
    )

    with open(img_folder / "turn3_raw.txt", "w") as f:
        f.write(text3)

    result = _safe_json_loads(text3)

    # ------------------------------------------------------------------
    # 8. Persist prediction + evaluation/visualisation
    # ------------------------------------------------------------------

    if task == "localization":
        ensure_evaluation_keys(result)
    preds.append(result)

    save_prediction(img_folder, result)
    save_reference(img_folder, batch_idx, hf_ds)

    if task == "localization":
        evaluate_prediction(img_folder, task)

    # ------------------------------------------------------------------
    # 9. Generate bbox visualization (for localization task)
    # ------------------------------------------------------------------

    if task == "localization":
        # Define the _draw_boxes helper function (same as in process_batch)
        def _draw_boxes(img_path: Path, gt: list[Any], pred: list[Any], out_path: Path):
            """Draw *gt* (green) and *pred* (red) boxes on *img_path* and save.

            The helper is now tolerant to various box formats:

            1. [x1, y1, x2, y2] - preferred.
            2. Dicts with keys (x1,y1,x2,y2) or (x,y,width,height).
            Invalid or incomplete entries are silently skipped.
            """
            import matplotlib

            matplotlib.use("Agg")  # headless
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt
            from matplotlib import patheffects as pe
            from PIL import Image

            img = Image.open(img_path).convert("L")
            fig, ax = plt.subplots(1, figsize=(6, 6))
            ax.imshow(img, cmap="gray")

            def _iter_boxes(raw_boxes):
                for b in raw_boxes:
                    if isinstance(b, (list, tuple)) and len(b) == 4:
                        yield b
                    elif isinstance(b, dict):
                        if all(k in b for k in ("x1", "y1", "x2", "y2")):
                            yield [b["x1"], b["y1"], b["x2"], b["y2"]]
                        elif all(k in b for k in ("x", "y", "width", "height")):
                            yield [b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]]

            for box in _iter_boxes(gt):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle(
                    (x1, y1), w, h, linewidth=1.5, edgecolor="lime", facecolor="none"
                )
                ax.add_patch(rect)

            for box in _iter_boxes(pred):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle(
                    (x1, y1), w, h, linewidth=1.2, edgecolor="red", facecolor="none", linestyle="--"
                )
                ax.add_patch(rect)

            ax.axis("off")

            # ------------------------------------------------------------------
            # Add legend (only for categories that are present)
            # ------------------------------------------------------------------
            from matplotlib.lines import Line2D  # imported here to keep the

            # function self-contained even when matplotlib is not globally
            # available at import time.

            legend_elements = []
            if gt:
                legend_elements.append(Line2D([0], [0], color="lime", lw=2, label="Ground Truth"))
            if pred:
                legend_elements.append(
                    Line2D([0], [0], color="red", lw=2, linestyle="--", label="Prediction")
                )

            if legend_elements:
                leg = ax.legend(
                    handles=legend_elements, loc="upper right", fontsize="x-small", frameon=False
                )
                # Improve readability - bright text with thin black outline
                for text in leg.get_texts():
                    text.set_color("yellow")
                    text.set_path_effects(
                        [
                            pe.Stroke(linewidth=1.0, foreground="black"),
                            pe.Normal(),
                        ]
                    )

            fig.tight_layout(pad=0)
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

        # Generate bbox visualization
        gt_bg = hf_ds[batch_idx].get("bbox_gold", {})
        gt_boxes = [
            [x, y, x + w, y + h]
            for x, y, w, h in zip(
                gt_bg.get("x", []),
                gt_bg.get("y", []),
                gt_bg.get("width", []),
                gt_bg.get("height", []),
            )
        ]
        viz_path = img_folder / "bboxes.png"
        _draw_boxes(img_path, gt_boxes, result.get("boxes", []), viz_path)

    # Calculate total tokens and cost
    total_tokens = log1.tokens + log2.tokens + log3.tokens
    total_cost = log1.cost + log2.cost + log3.cost

    # Add costs from optional steps
    if clinical_history_integration:
        total_tokens += log2b.tokens
        total_cost += log2b.cost

    # Add costs from iterative retrieval
    for i in range(request_count):
        # We don't have access to individual log objects for retrieval requests
        # This is an approximation
        total_tokens += 500  # Approximate tokens per retrieval request
        total_cost += 0.001  # Approximate cost per retrieval request

    logger.info(
        "[multiturn] Image %d processed - tokens: %.0f, cost: $%.4f, retrieval_requests: %d",
        batch_idx,
        total_tokens,
        total_cost,
        request_count,
    )

    return result


def process_batch_visual_multiturn(
    batch_idx: int,
    batch: dict,
    main_run_dir: Path,
    task: str,
    cfg: Config,
    adapter,
    retriever,
    preds: list,
    hf_ds,
):
    """Multi-turn pipeline with visual adjustment and retrieval."""

    logger.debug("[visual_multiturn] Starting workflow for image %d", batch_idx)

    from PIL import Image

    hf_record = hf_ds[batch_idx]

    if "image" in hf_record and hf_record["image"] is not None:
        pil = hf_record["image"]
        if not isinstance(pil, Image.Image):
            pil = Image.fromarray(pil)
        pil = pil.convert("L")
    else:
        img_path_str = hf_record.get("image_path")
        if not img_path_str:
            raise ValueError(f"No image or image_path field for record {batch_idx}")
        pil = Image.open(img_path_str).convert("L")

    img_folder = main_run_dir / f"image_{batch_idx}"
    img_folder.mkdir(parents=True, exist_ok=True)
    img_path = img_folder / "image.png"
    pil.save(img_path)

    processed_path = img_path
    for i in range(max(cfg.visual_rounds, 1)):
        prompt_ops = load_prompt(
            "visual_multiturn/ops_request.jinja",
            img_path=processed_path,
            passages=[],
            metadata={"image_id": batch_idx},
        )
        text_ops, _ = asyncio.run(
            adapter.generate(processed_path, passages=[], system_prompt=prompt_ops)
        )
        with open(img_folder / f"ops_round_{i+1}.txt", "w") as f:
            f.write(text_ops)

        def _safe_json_loads(payload: str):
            import json as _json
            import re

            try:
                return _json.loads(payload)
            except _json.JSONDecodeError:
                m = re.search(r"\{.*\}", payload, re.DOTALL)
                if m:
                    try:
                        return _json.loads(m.group(0))
                    except _json.JSONDecodeError:
                        pass
            logger.warning("[visual_multiturn] Could not parse JSON for ops round %d", i + 1)
            return {}

        ops = _safe_json_loads(text_ops)
        img = Image.open(processed_path)
        if zf := ops.get("zoom_factor"):
            img = zoom_image(img, float(zf))
        if cb := ops.get("crop_box"):
            img = crop_image(img, tuple(map(int, cb)))
        if cf := ops.get("contrast_factor"):
            img = adjust_contrast(img, float(cf))
        if ir := ops.get("intensity_range"):
            low, high = map(int, ir)
            img = apply_intensity_threshold(img, low, high)

        processed_path = img_folder / f"processed_{i+1}.png"
        img.save(processed_path)

        if not ops.get("need_more_ops", False):
            break

    hf_record_local = dict(hf_record)
    hf_record_local["image_path"] = str(processed_path)
    tmp_ds = [hf_record_local for _ in range(batch_idx + 1)]

    result = process_batch_multiturn(
        batch_idx=batch_idx,
        batch=batch,
        main_run_dir=main_run_dir,
        task=task,
        cfg=cfg,
        adapter=adapter,
        retriever=retriever,
        preds=preds,
        hf_ds=tmp_ds,
    )

    return result


if __name__ == "__main__":
    main()
