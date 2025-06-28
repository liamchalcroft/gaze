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
from nova_retrieval_vlm.guidelines.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    CrossEncoderReranker,
    MedicalQueryExpander,
)
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.prompts.enhanced_prompt_loader import load_enhanced_prompt, get_system_prompt
from nova_retrieval_vlm.visual_reasoning.image_ops import (
    adjust_contrast,
    apply_intensity_threshold,
    crop_image,
    zoom_image,
)
from nova_retrieval_vlm.retrieval.web_search import MedicalWebSearcher

# ---------------------------------------------------------------------------
# Robust JSON parsing helper (shared by all pipelines)
# ---------------------------------------------------------------------------

import re
from ast import literal_eval


def robust_json_loads(payload: str):  # noqa: D401
    """Best-effort JSON deserialization.

    Models occasionally wrap their JSON answer in Markdown fences, add a
    leading *Answer:* prefix, or use single quotes / trailing commas.  This
    helper applies a series of increasingly permissive strategies and only
    gives up as a *last resort*.  The returned object is **always** a dict so
    that downstream evaluation code does not crash.
    """

    import json

    # Strategy 1 — direct parse ------------------------------------------------
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Strategy 2 — strip common wrappers --------------------------------------
    cleaned = payload.strip()
    cleaned = re.sub(r"^Answer:\s*", "", cleaned, flags=re.IGNORECASE)
    # Remove ```json ... ``` or generic ``` fenced blocks
    cleaned = re.sub(r"```(?:json)?", "", cleaned)
    cleaned = cleaned.strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Strategy 3 — extract first {...} block -----------------------------------
    m = re.search(r"\{.*\}", payload, flags=re.DOTALL)
    if m:
        snippet = m.group(0)
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Strategy 4 — literal_eval for Python-style dicts -------------------------
    try:
        obj = literal_eval(snippet if 'snippet' in locals() else payload)
        if isinstance(obj, dict):
            return obj
    except Exception:  # pylint: disable=broad-except
        pass

    # Final fallback -----------------------------------------------------------
    from loguru import logger as _logger

    _logger.warning("robust_json_loads: failed to decode Model output – returning stub")
    return {"raw": payload}

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
            reranker = None
            query_expander = None
            # Enable re-ranking if sentence-transformers CrossEncoder available
            try:
                reranker = CrossEncoderReranker()
            except ImportError:
                logger.warning("CrossEncoder not available – skipping re-ranking stage")
            query_expander = MedicalQueryExpander()
            return HybridRetriever(
                BM25Retriever(str(bm25_idx)),
                DenseRetriever(str(faiss_idx)),
                alpha=cfg.retrieval.hybrid_ratio,
                reranker=reranker,
                query_expander=query_expander,
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

    passages: list[str] = []
    template_name = f"baseline/{task}.jinja"
    retrieval_debug_info = None

    # ------------------------------------------------------------------
    # Safe access to optional metadata list.  Some NOVA records have an
    # empty list or omit the field altogether, which previously caused
    # IndexError / KeyError when we assumed element [0] existed.
    # ------------------------------------------------------------------

    _meta_list = batch.get("metadata", []) or []
    if isinstance(_meta_list, (list, tuple)) and _meta_list and isinstance(_meta_list[0], dict):
        metadata_rec = _meta_list[0]
    else:
        metadata_rec = {}

    if cfg.use_retrieval and retriever:
        query = (
            metadata_rec.get("final_diagnosis")
            or metadata_rec.get("diagnosis")
            or metadata_rec.get("caption")
            or metadata_rec.get("clinical_history")
            or ""
        )
        retrieval_debug_info = {"query": query, "success": False, "error": None, "passages": []}
        try:
            passages = retriever(query, k=cfg.retrieval.top_k) if query else []
            retrieval_debug_info["success"] = True
            retrieval_debug_info["passages"] = passages
            template_name = f"retrieval_{task}.jinja"
        except Exception as exc:
            logger.warning("[baseline] Retrieval failed: %s", exc)
            retrieval_debug_info["error"] = str(exc)

    # Prepare metadata for the prompt, always including image_id and dims
    metadata_for_prompt = {
        "image_id": batch_idx,
        "width": pil.width,
        "height": pil.height,
    }
    if task == "diagnosis":
        # Safe access to optional metadata list
        _meta_list = batch.get("metadata", []) or []
        if isinstance(_meta_list, (list, tuple)) and _meta_list and isinstance(_meta_list[0], dict):
            metadata_for_prompt["clinical_history"] = _meta_list[0].get("clinical_history", "")
        else:
            metadata_for_prompt["clinical_history"] = ""

    # Load prompt with enhanced system prompt integration
    prompt = load_enhanced_prompt(
        template_name=template_name,
        image_path=img_path,
        passages=passages,
        metadata=metadata_for_prompt,
    )
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

    if retrieval_debug_info:
        with open(img_folder / "retrieval_debug.txt", "w") as f:
            f.write(f"Query: {retrieval_debug_info['query']}\n")
            f.write(f"Success: {retrieval_debug_info['success']}\n")
            if retrieval_debug_info.get("error"):
                f.write(f"Error: {retrieval_debug_info['error']}\n")
            f.write("\n=== PASSAGES ===\n")
            for p in passages:
                f.write(p + "\n\n")

    # -------------------------------------------------------------
    # Parse JSON with robustness to badly-formatted model output
    # -------------------------------------------------------------

    result = robust_json_loads(text)

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
    """Process a single batch using multi-turn reasoning approach."""
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

    # Prepare metadata for the prompt
    metadata_for_prompt = {
        "image_id": batch_idx,
        "width": pil.width,
        "height": pil.height,
    }
    if task == "diagnosis":
        # Safe access to optional metadata list
        _meta_list = batch.get("metadata", []) or []
        if isinstance(_meta_list, (list, tuple)) and _meta_list and isinstance(_meta_list[0], dict):
            metadata_for_prompt["clinical_history"] = _meta_list[0].get("clinical_history", "")
        else:
            metadata_for_prompt["clinical_history"] = ""

    # Step 1: Initial observations and differential diagnosis
    logger.info(f"Multi-turn Step 1 for image {batch_idx}")
    step1_prompt = load_enhanced_prompt(
        template_name="multiturn/step1.jinja",
        image_path=img_path,
        passages=[],
        metadata=metadata_for_prompt,
    )
    
    with open(img_folder / "step1_prompt.txt", "w") as f:
        f.write(step1_prompt)
    
    step1_text, step1_log = asyncio.run(adapter.generate(img_path, [], system_prompt=step1_prompt))
    logger.info(f"Step 1 generation cost: {step1_log}")
    
    with open(img_folder / "step1_output.txt", "w") as f:
        f.write(step1_text)
    
    step1_result = robust_json_loads(step1_text)
    
    # Step 2: Comprehensive analysis with guidelines
    logger.info(f"Multi-turn Step 2 for image {batch_idx}")
    
    # Prepare retrieval query based on step 1 findings
    passages = []
    retrieval_debug_info = None
    if cfg.use_retrieval and retriever:
        # Use step 1 findings to create a targeted query
        query_parts = []
        if step1_result.get("differential"):
            query_parts.extend(step1_result["differential"][:3])  # Top 3 diagnoses
        if step1_result.get("observations"):
            query_parts.append(step1_result["observations"])
        
        query = " ".join(query_parts) if query_parts else ""
        retrieval_debug_info = {"query": query, "success": False, "error": None, "passages": []}
        
        try:
            passages = retriever(query, k=cfg.retrieval.top_k) if query else []
            retrieval_debug_info["success"] = True
            retrieval_debug_info["passages"] = passages
        except Exception as exc:
            logger.warning("[multiturn] Retrieval failed: %s", exc)
            retrieval_debug_info["error"] = str(exc)
    
    step2_prompt = load_enhanced_prompt(
        template_name="multiturn/step2.jinja",
        image_path=img_path,
        passages=passages,
        metadata=metadata_for_prompt,
    )
    
    with open(img_folder / "step2_prompt.txt", "w") as f:
        f.write(step2_prompt)
    
    step2_text, step2_log = asyncio.run(adapter.generate(img_path, passages, system_prompt=step2_prompt))
    logger.info(f"Step 2 generation cost: {step2_log}")
    
    with open(img_folder / "step2_output.txt", "w") as f:
        f.write(step2_text)
    
    # Step 3: Final task-specific output
    logger.info(f"Multi-turn Step 3 for image {batch_idx}")
    
    # Determine the appropriate step 3 template based on task
    step3_template_map = {
        "caption": "multiturn/caption_step3.jinja",
        "diagnosis": "multiturn/diagnosis_step3.jinja", 
        "localization": "multiturn/localization_step3.jinja",
    }
    
    step3_template = step3_template_map.get(task, "multiturn/caption_step3.jinja")
    
    # Add step 1 and step 2 results to metadata for step 3
    step3_metadata = metadata_for_prompt.copy()
    step3_metadata["step1_result"] = step1_result
    step3_metadata["step2_analysis"] = step2_text
    
    step3_prompt = load_enhanced_prompt(
        template_name=step3_template,
        image_path=img_path,
        passages=passages,
        metadata=step3_metadata,
    )
    
    with open(img_folder / "step3_prompt.txt", "w") as f:
        f.write(step3_prompt)
    
    step3_text, step3_log = asyncio.run(adapter.generate(img_path, passages, system_prompt=step3_prompt))
    logger.info(f"Step 3 generation cost: {step3_log}")
    
    with open(img_folder / "step3_output.txt", "w") as f:
        f.write(step3_text)
    
    # Parse final result
    result = robust_json_loads(step3_text)
    
    # Add metadata
    result["image_path"] = str(img_path)
    result["ground_truth_image_idx"] = batch_idx
    result["multiturn_steps"] = {
        "step1": step1_result,
        "step2_analysis": step2_text,
        "step3": step3_text,
    }
    
    # Ensure required keys are present
    ensure_evaluation_keys(result)
    
    logger.info(f"Successfully processed multi-turn result for image {batch_idx}")
    
    # Save individual prediction for this image
    save_prediction(img_folder, result)
    
    # Generate and save reference
    save_reference(img_folder, batch_idx, hf_ds)
    
    # Save retrieval debug info if available
    if retrieval_debug_info:
        with open(img_folder / "retrieval_debug.txt", "w") as f:
            f.write(f"Query: {retrieval_debug_info['query']}\n")
            f.write(f"Success: {retrieval_debug_info['success']}\n")
            if retrieval_debug_info.get("error"):
                f.write(f"Error: {retrieval_debug_info['error']}\n")
            f.write("\n=== PASSAGES ===\n")
            for p in passages:
                f.write(p + "\n\n")
    
    preds.append(result)


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
    """Process a single batch using visual multi-turn reasoning approach."""
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

    # Prepare metadata for the prompt
    metadata_for_prompt = {
        "image_id": batch_idx,
        "width": pil.width,
        "height": pil.height,
    }
    if task == "diagnosis":
        # Safe access to optional metadata list
        _meta_list = batch.get("metadata", []) or []
        if isinstance(_meta_list, (list, tuple)) and _meta_list and isinstance(_meta_list[0], dict):
            metadata_for_prompt["clinical_history"] = _meta_list[0].get("clinical_history", "")
        else:
            metadata_for_prompt["clinical_history"] = ""

    # Initialize web searcher for visual multi-turn
    web_searcher = MedicalWebSearcher() if cfg.approach == "visual_multiturn" else None
    
    # Visual multi-turn analysis with multiple rounds
    current_image_path = img_path
    analysis_notes = []
    visual_operations = []
    web_search_results = []
    
    for round_num in range(cfg.visual_rounds):
        logger.info(f"Visual multi-turn round {round_num + 1} for image {batch_idx}")
        
        # Load visual operations request prompt
        ops_prompt = load_enhanced_prompt(
            template_name="visual_multiturn/ops_request.jinja",
            image_path=current_image_path,
            passages=[],
            metadata={
                **metadata_for_prompt,
                "round": round_num + 1,
                "total_rounds": cfg.visual_rounds,
                "analysis_notes": "\n".join(analysis_notes) if analysis_notes else "None",
                "web_search_results": "\n".join(web_search_results) if web_search_results else "None",
            },
        )
        
        with open(img_folder / f"round_{round_num + 1}_ops_prompt.txt", "w") as f:
            f.write(ops_prompt)
        
        ops_text, ops_log = asyncio.run(adapter.generate(current_image_path, [], system_prompt=ops_prompt))
        logger.info(f"Round {round_num + 1} operations generation cost: {ops_log}")
        
        with open(img_folder / f"round_{round_num + 1}_ops_output.txt", "w") as f:
            f.write(ops_text)
        
        ops_result = robust_json_loads(ops_text)
        
        # Apply visual operations if requested
        if ops_result.get("visual_operations"):
            for op in ops_result["visual_operations"]:
                try:
                    if op.get("operation") == "zoom":
                        current_image_path = zoom_image(
                            current_image_path, 
                            op.get("region", [0, 0, pil.width, pil.height]),
                            op.get("factor", 2.0)
                        )
                    elif op.get("operation") == "crop":
                        current_image_path = crop_image(
                            current_image_path,
                            op.get("region", [0, 0, pil.width, pil.height])
                        )
                    elif op.get("operation") == "contrast":
                        current_image_path = adjust_contrast(
                            current_image_path,
                            op.get("factor", 1.5)
                        )
                    elif op.get("operation") == "threshold":
                        current_image_path = apply_intensity_threshold(
                            current_image_path,
                            op.get("min_threshold", 0),
                            op.get("max_threshold", 255)
                        )
                    
                    visual_operations.append(op)
                except Exception as e:
                    logger.warning(f"Visual operation failed: {e}")
        
        # Perform web search if requested
        if ops_result.get("web_search_requests") and web_searcher:
            for search_req in ops_result["web_search_requests"]:
                try:
                    search_type = search_req.get("type", "general")
                    query = search_req.get("query", "")
                    
                    if search_type == "medical":
                        results = web_searcher.medical_search(query)
                    else:
                        results = web_searcher.general_search(query)
                    
                    web_search_results.append(f"Query: {query}\nResults: {results}")
                except Exception as e:
                    logger.warning(f"Web search failed: {e}")
                    web_search_results.append(f"Query: {query}\nError: {str(e)}")
        
        # Add analysis notes
        if ops_result.get("analysis_notes"):
            analysis_notes.append(f"Round {round_num + 1}: {ops_result['analysis_notes']}")
    
    # Final analysis using original image
    logger.info(f"Final analysis for image {batch_idx}")
    
    # Determine the appropriate final template based on task
    final_template_map = {
        "caption": "baseline/caption.jinja",
        "diagnosis": "baseline/diagnosis.jinja",
        "localization": "baseline/localization.jinja",
    }
    
    final_template = final_template_map.get(task, "baseline/caption.jinja")
    
    # Add visual analysis context to metadata
    final_metadata = metadata_for_prompt.copy()
    final_metadata["visual_operations"] = visual_operations
    final_metadata["analysis_notes"] = "\n".join(analysis_notes)
    final_metadata["web_search_results"] = "\n".join(web_search_results)
    
    final_prompt = load_enhanced_prompt(
        template_name=final_template,
        image_path=img_path,  # Use original image for final analysis
        passages=[],
        metadata=final_metadata,
    )
    
    with open(img_folder / "final_prompt.txt", "w") as f:
        f.write(final_prompt)
    
    final_text, final_log = asyncio.run(adapter.generate(img_path, [], system_prompt=final_prompt))
    logger.info(f"Final analysis generation cost: {final_log}")
    
    with open(img_folder / "final_output.txt", "w") as f:
        f.write(final_text)
    
    # Parse final result
    result = robust_json_loads(final_text)
    
    # Add metadata
    result["image_path"] = str(img_path)
    result["ground_truth_image_idx"] = batch_idx
    result["visual_multiturn_data"] = {
        "visual_operations": visual_operations,
        "analysis_notes": analysis_notes,
        "web_search_results": web_search_results,
    }
    
    # Ensure required keys are present
    ensure_evaluation_keys(result)
    
    logger.info(f"Successfully processed visual multi-turn result for image {batch_idx}")
    
    # Save individual prediction for this image
    save_prediction(img_folder, result)
    
    # Generate and save reference
    save_reference(img_folder, batch_idx, hf_ds)
    
    preds.append(result)