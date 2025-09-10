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

# ---------------------------------------------------------------------------
# Structured response parsing
# ---------------------------------------------------------------------------
from nova_retrieval_vlm.config import Config
from nova_retrieval_vlm.data.nova_dataset import get_dataloader
from nova_retrieval_vlm.evaluation import evaluate
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.prompts.enhanced_prompt_loader import load_jinja_template_prompt
from nova_retrieval_vlm.retrieval.retrievers import BM25Retriever
from nova_retrieval_vlm.retrieval.retrievers import CrossEncoderReranker
from nova_retrieval_vlm.retrieval.retrievers import DenseRetriever
from nova_retrieval_vlm.retrieval.retrievers import HybridRetriever
from nova_retrieval_vlm.retrieval.retrievers import MedicalQueryExpander
from nova_retrieval_vlm.retrieval.web_search import MedicalWebSearcher
from nova_retrieval_vlm.utils.batch_processing_utils import BatchContext
from nova_retrieval_vlm.utils.batch_processing_utils import postprocess_batch_result
from nova_retrieval_vlm.visual_reasoning.image_ops import adjust_contrast
from nova_retrieval_vlm.visual_reasoning.image_ops import apply_intensity_threshold
from nova_retrieval_vlm.visual_reasoning.image_ops import crop_image
from nova_retrieval_vlm.visual_reasoning.image_ops import zoom_image


class JSONParseError(Exception):
    """Raised when JSON parsing fails definitively."""

    def __init__(self, original_content: str, error: str) -> None:
        self.original_content = original_content
        self.error = error
        super().__init__(f"Failed to parse JSON: {error}")


def parse_model_json_response(payload: str) -> dict[str, Any]:
    """Parse JSON response from model with minimal preprocessing.

    Args:
        payload: Raw response string from model

    Returns:
        Parsed dictionary

    Raises:
        JSONParseError: If parsing fails after basic cleanup
    """
    # Minimal cleanup - remove markdown fences and common prefixes
    cleaned = payload.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    if cleaned.lower().startswith("answer:"):
        cleaned = cleaned[7:]

    cleaned = cleaned.strip()

    try:
        parsed_response = json.loads(cleaned)
        if not isinstance(parsed_response, dict):
            raise JSONParseError(payload, f"Expected dict, got {type(parsed_response)}")
        return parsed_response
    except json.JSONDecodeError as e:
        raise JSONParseError(payload, f"JSON decode error: {e}") from e


def extract_batch_metadata(batch: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata from batch with proper validation.

    Args:
        batch: Batch dictionary containing metadata

    Returns:
        Metadata dictionary

    Raises:
        ValueError: If metadata is missing or invalid
    """
    if "metadata" not in batch:
        raise ValueError("Batch missing required 'metadata' field")

    meta_list = batch["metadata"]
    if not isinstance(meta_list, list) or not meta_list:
        raise ValueError("Metadata must be non-empty list")

    metadata = meta_list[0]
    if not isinstance(metadata, dict):
        raise ValueError("Metadata items must be dictionaries")

    return metadata
    return {}


def validate_environment() -> None:
    """Validate required environment variables are set.

    Raises:
        EnvironmentError: If required environment variables are missing
    """
    required_vars = []

    # Check for API keys - need at least one
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        required_vars.append("OPENROUTER_API_KEY or OPENAI_API_KEY")

    # Check for data directory if set
    data_dir = os.getenv("DATA_DIR")
    if data_dir and not Path(data_dir).exists():
        logger.warning(
            f"DATA_DIR environment variable points to non-existent directory: {data_dir}"
        )

    # Check for output directory if set
    output_dir = os.getenv("OUTPUT_DIR")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created output directory: {output_dir}")

    if required_vars:
        raise OSError(
            f"Missing required environment variables: {', '.join(required_vars)}\n"
            f"Please set them in your .env file or environment.\n"
            f"See the README or docs/usage.md for setup instructions."
        )

    logger.debug("Environment validation passed")


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


def _parse_cli_args():
    """Parse command line arguments manually and return a Config object."""
    config = Config()

    # Parse arguments in the format key=value
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)

            # Handle nested keys like model.name
            if "." in key:
                parts = key.split(".")
                current = config
                for part in parts[:-1]:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        sys.exit(1)
                setattr(current, parts[-1], value)
            elif hasattr(config, key):
                # Convert value to appropriate type
                current_value = getattr(config, key)
                if isinstance(current_value, bool):
                    setattr(config, key, value.lower() in ("true", "1", "yes", "on"))
                elif isinstance(current_value, int):
                    setattr(config, key, int(value))
                elif isinstance(current_value, float):
                    setattr(config, key, float(value))
                else:
                    setattr(config, key, value)
            else:
                sys.exit(1)

    return config


def _validate_config(config: Config) -> None:
    """Validate configuration and provide helpful error messages."""

    # Available options
    available_approaches = [
        "baseline",
        "multiturn",
        "visual",
        "retrieval",
        "web_search",
        "comprehensive",
    ]
    available_tasks = ["caption", "diagnosis", "localization", "visualize", "prompt"]

    # Validate approach
    if config.approach not in available_approaches:
        raise ValueError(
            f"Invalid approach: '{config.approach}'. "
            f"Available approaches: {', '.join(available_approaches)}\n"
            f"Use one of the optimized system prompt modes for best performance."
        )

    # Validate task
    if config.task not in available_tasks:
        raise ValueError(
            f"Invalid task: '{config.task}'. Available tasks: {', '.join(available_tasks)}"
        )

    # Validate approach-specific requirements
    if config.approach == "retrieval" and not config.use_retrieval:
        logger.warning(
            "Approach 'retrieval' requires retrieval to be enabled. "
            "Setting use_retrieval=True automatically."
        )
        config.use_retrieval = True

    if config.approach == "web_search" and not config.use_web_search:
        logger.warning(
            "Approach 'web_search' requires web search to be enabled. "
            "Setting use_web_search=True automatically."
        )
        config.use_web_search = True

    if config.approach == "comprehensive":
        if not config.use_retrieval:
            logger.warning(
                "Comprehensive approach works best with retrieval enabled. "
                "Consider setting use_retrieval=True for optimal performance."
            )
        if not config.use_web_search:
            logger.warning(
                "Comprehensive approach works best with web search enabled. "
                "Consider setting use_web_search=True for optimal performance."
            )

    # Validate model configuration
    if not config.model.name:
        raise ValueError("Model name is required. Set model.name to a valid model.")

    # Validate batch size
    if config.batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    # Validate retrieval configuration
    if config.use_retrieval:
        # Convert string to int if needed (Hydra sometimes passes as string)
        if isinstance(config.retrieval.top_k, str):
            try:
                config.retrieval.top_k = int(config.retrieval.top_k)
            except ValueError as e:
                raise ValueError(
                    f"Retrieval top_k must be a valid integer, got: {config.retrieval.top_k}"
                ) from e

        if config.retrieval.top_k <= 0:
            raise ValueError("Retrieval top_k must be positive when retrieval is enabled.")

    # Validate visual configuration
    if config.approach == "visual":
        # Convert string to int if needed
        if isinstance(config.visual_rounds, str):
            try:
                config.visual_rounds = int(config.visual_rounds)
            except ValueError as e:
                raise ValueError(
                    f"Visual rounds must be a valid integer, got: {config.visual_rounds}"
                ) from e

        if config.visual_rounds <= 0:
            raise ValueError("Visual rounds must be positive for visual approach.")

    # Validate multiturn configuration
    if config.approach == "multiturn":
        # Convert string to int if needed
        if isinstance(config.multiturn_max_steps, str):
            try:
                config.multiturn_max_steps = int(config.multiturn_max_steps)
            except ValueError as e:
                raise ValueError(
                    f"Multiturn max steps must be a valid integer, got: {config.multiturn_max_steps}"
                ) from e

        if config.multiturn_max_steps <= 0:
            raise ValueError("Multiturn max steps must be positive for multiturn approach.")

    # Validate comprehensive timeout
    if config.approach == "comprehensive":
        # Convert string to int if needed
        if isinstance(config.comprehensive_timeout, str):
            try:
                config.comprehensive_timeout = int(config.comprehensive_timeout)
            except ValueError as e:
                raise ValueError(
                    f"Comprehensive timeout must be a valid integer, got: {config.comprehensive_timeout}"
                ) from e

        if config.comprehensive_timeout <= 0:
            raise ValueError("Comprehensive timeout must be positive for comprehensive approach.")

    # Log configuration summary
    logger.info(f"🚀 Starting analysis with approach='{config.approach}' and task='{config.task}'")

    if config.approach == "baseline":
        logger.info("📋 Using baseline approach with enhanced JSON format specification")
    elif config.approach == "multiturn":
        logger.info(f"🔄 Using multiturn approach with max {config.multiturn_max_steps} steps")
    elif config.approach == "visual":
        logger.info(f"👁️ Using visual approach with {config.visual_rounds} rounds")
    elif config.approach == "retrieval":
        logger.info(f"📚 Using retrieval approach with top_k={config.retrieval.top_k}")
    elif config.approach == "web_search":
        logger.info("🌐 Using web search approach with query formulation")
    elif config.approach == "comprehensive":
        logger.info("🎯 Using comprehensive approach with all capabilities")

    if config.use_retrieval:
        logger.info(
            f"📖 Retrieval enabled: {config.retrieval.type} with top_k={config.retrieval.top_k}"
        )

    if config.use_web_search:
        logger.info("🔍 Web search enabled for current medical information")


@hydra.main(version_base="1.1", config_path=None, config_name="config")
def main(config: Config) -> None:
    """
    Main entry point for running experiments with enhanced system prompts.

    Available approaches (optimized system prompts):
      - baseline: Standard single-turn analysis with enhanced JSON format
      - multiturn: Iterative multi-step analysis with conditional continuation
      - visual: Visual operations and analysis with enhanced guidance
      - retrieval: Knowledge-augmented analysis with medical literature
      - web_search: Web search-augmented analysis with query formulation
      - comprehensive: All capabilities combined with performance optimization

    Available tasks:
      - caption: Generate detailed medical image descriptions
      - diagnosis: Provide differential diagnoses with confidence levels
      - localization: Identify and localize anatomical structures/abnormalities

    Usage examples:
      # Baseline analysis
      python -m nova_retrieval_vlm.cli approach=baseline task=caption model.name=qwen-vl-chat batch_size=4

      # Multi-turn analysis with retrieval
      python -m nova_retrieval_vlm.cli approach=multiturn task=diagnosis use_retrieval=true retrieval.top_k=5

      # Visual operations analysis
      python -m nova_retrieval_vlm.cli approach=visual task=localization visual_rounds=3

      # Retrieval-augmented analysis
      python -m nova_retrieval_vlm.cli approach=retrieval task=diagnosis retrieval.type=hybrid

      # Web search-augmented analysis
      python -m nova_retrieval_vlm.cli approach=web_search task=caption use_web_search=true

      # Comprehensive analysis with all capabilities
      python -m nova_retrieval_vlm.cli approach=comprehensive task=diagnosis use_retrieval=true use_web_search=true
    """
    # Convert OmegaConf to our Config dataclass for better type safety
    if isinstance(config, OmegaConf):
        # Create a new Config instance from the OmegaConf
        config_dict = OmegaConf.to_container(config, resolve=True)
        config = Config(**config_dict)

    logger.add(lambda msg: print(msg, end=""), level="INFO")  # noqa: T201 - Required for CLI logging

    # Validate configuration
    _validate_config(config)

    # Validate environment variables
    validate_environment()

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    data_dir = to_absolute_path(config.paths.data_dir)
    output_dir = to_absolute_path(config.paths.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Handle free-form prompt-only task
    if config.task == "prompt":
        adapter = create_model_adapter(config)
        text, log = asyncio.run(adapter.generate_text(config.prompt_text))
        logger.info(f"Generation cost: {log}")
        return

    # Handle visualization task and exit early
    if config.task == "visualize":
        from nova_retrieval_vlm.visualization.sample_utils import visualize_samples

        visualize_samples(
            num_samples=config.visualization.num_samples,
            out_dir=output_dir,
            cache_dir=data_dir,
            trust_remote_code=config.visualization.trust_remote_code,
            overlay=config.visualization.overlay,
        )
        return

    # Load data (always test split zero-shot)
    dl = get_dataloader(
        batch_size=config.batch_size,
        data_dir=data_dir,
    )
    task = config.task  # 'baseline','retrieval','localization','caption','diagnosis'

    # Setup retriever if retrieval augmentation is enabled
    retriever = create_retriever_instance(config)

    # Setup adapter with fallback support
    adapter = create_model_adapter(config)

    preds: list[dict] = []
    # Iterate over dataset
    # ---- Set iteration limit for testing or full dataset ----
    max_iterations = config.max_iterations if config.max_iterations > 0 else float("inf")
    current_iteration = 0

    # -------------------------------------------------------------
    # Determine (or create) the main run directory.  We support three
    # scenarios:
    #   1. `resume_dir` explicitly provided → use that directory.
    #   2. No explicit path, but an *existing* timestamped directory is
    #      present under `output_dir` and `skip_existing=True` → automatically
    #      resume in the *latest* directory.
    #   3. Fresh run → create a new timestamped directory.
    # -------------------------------------------------------------

    if config.resume_dir:
        main_run_dir = Path(to_absolute_path(config.resume_dir))
        if not main_run_dir.exists():
            raise FileNotFoundError(
                f"resume_dir={config.resume_dir} does not exist – unable to resume."
            )
        logger.info(f"🔄 Resuming in user-specified run directory: {main_run_dir}")
    else:
        # Auto-resume heuristic – pick the most recent numeric sub-directory
        # (created via create_run_directory) if the user asked to skip
        # existing samples.
        candidates = (
            [p for p in Path(output_dir).iterdir() if p.is_dir() and p.name.isdigit()]
            if Path(output_dir).exists()
            else []
        )

        if config.skip_existing and candidates:
            main_run_dir = sorted(candidates, key=lambda p: int(p.name))[-1]
            logger.info(f"🔄 Auto-resuming in latest run directory: {main_run_dir}")
        else:
            main_run_dir = create_run_directory(output_dir)

    # ---------------------------------------------------------------------
    # Reference dataset (needed for creating refs.jsonl)
    # ---------------------------------------------------------------------

    local_path = Path(to_absolute_path(config.paths.data_dir)) / "nova_test"
    logger.info(f"Looking for pre-processed arrow dataset under: {local_path}")

    if local_path.exists():
        # Preferred fast-path - load the arrow file we previously generated
        huggingface_dataset = load_dataset(
            "arrow", data_files=str(local_path / "data-00000-of-00001.arrow")
        )["train"]
        logger.info("Loaded cached arrow dataset with %d samples", len(huggingface_dataset))
    else:
        # Fallback: use the HuggingFace dataset that was already downloaded by
        # NovaDataset (exposed via dl.dataset.hf_dataset).
        logger.warning("Cached arrow dataset not found - falling back to in-memory HF dataset.")
        # dl.dataset is an instance of NovaDataset; we expose the underlying HF
        # dataset via the public property `hf_dataset`.
        try:
            huggingface_dataset = dl.dataset.hf_dataset  # type: ignore[attr-defined]
            logger.info(
                "Using HF dataset from NovaDataset with %d samples", len(huggingface_dataset)
            )
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

        # -------------------------------------------------------------
        # Skip if prediction already exists (resumable runs)
        # -------------------------------------------------------------

        if _skip_if_existing(batch_idx, main_run_dir, config):
            logger.info(f"⏭️  Skipping image {batch_idx} – existing prediction found (resume mode).")
            continue

        # Process each batch according to the chosen *approach*
        if config.approach == "baseline":
            process_batch_baseline(
                batch_idx=batch_idx,
                batch=batch,
                main_run_dir=main_run_dir,
                task=task,
                config=config,
                adapter=adapter,
                retriever=retriever,
                preds=preds,
                huggingface_dataset=huggingface_dataset,
            )
        elif config.approach == "multiturn":
            process_batch_multiturn(
                batch_idx=batch_idx,
                batch=batch,
                main_run_dir=main_run_dir,
                task=task,
                config=config,
                adapter=adapter,
                retriever=retriever,
                preds=preds,
                huggingface_dataset=huggingface_dataset,
            )
        elif config.approach == "visual":
            process_batch_visual_multiturn(
                batch_idx=batch_idx,
                batch=batch,
                main_run_dir=main_run_dir,
                task=task,
                config=config,
                adapter=adapter,
                retriever=retriever,
                preds=preds,
                huggingface_dataset=huggingface_dataset,
            )
        elif config.approach == "retrieval":
            process_batch_retrieval(
                batch_idx=batch_idx,
                batch=batch,
                main_run_dir=main_run_dir,
                task=task,
                config=config,
                adapter=adapter,
                retriever=retriever,
                preds=preds,
                huggingface_dataset=huggingface_dataset,
            )
        elif config.approach == "web_search":
            process_batch_web_search(
                batch_idx=batch_idx,
                batch=batch,
                main_run_dir=main_run_dir,
                task=task,
                config=config,
                adapter=adapter,
                retriever=retriever,
                preds=preds,
                huggingface_dataset=huggingface_dataset,
            )
        elif config.approach == "comprehensive":
            process_batch_comprehensive(
                batch_idx=batch_idx,
                batch=batch,
                main_run_dir=main_run_dir,
                task=task,
                config=config,
                adapter=adapter,
                retriever=retriever,
                preds=preds,
                huggingface_dataset=huggingface_dataset,
            )
        # Legacy support for old approach names
        elif config.approach == "visual_multiturn":
            logger.warning(
                "approach='visual_multiturn' is deprecated, use approach='visual' instead"
            )
            process_batch_visual_multiturn(
                batch_idx=batch_idx,
                batch=batch,
                main_run_dir=main_run_dir,
                task=task,
                config=config,
                adapter=adapter,
                retriever=retriever,
                preds=preds,
                huggingface_dataset=huggingface_dataset,
            )
        else:
            available_approaches = [
                "baseline",
                "multiturn",
                "visual",
                "retrieval",
                "web_search",
                "comprehensive",
            ]
            raise ValueError(
                f"Unknown approach: {config.approach}. Available approaches: {available_approaches}"
            )

        # Add a delay to help with rate limiting
        time.sleep(config.request_delay)

        # Increment counter
        current_iteration += 1

    logger.info(f"Finished processing {len(preds)} predictions in this session.")

    if len(preds) == 0:
        logger.info("No new predictions generated – skipping evaluation and summary file creation.")
        return {"run_dir": str(main_run_dir), "metrics": {}}

    # Save predictions directly to main_run_dir (fixes auto-resume issue)
    preds_file = main_run_dir / "preds.jsonl"
    logger.info(f"Saving predictions to: {preds_file}")
    with open(preds_file, "w") as fw:
        for pred in preds:
            # Standardize for evaluation
            if isinstance(pred, dict):
                # Convert new localization schema to legacy format if needed
                if "localizations" in pred and "boxes" not in pred:
                    localizations = pred["localizations"]
                    boxes = []
                    labels = []
                    scores = []

                    for loc in localizations:
                        if "bounding_box" in loc:
                            boxes.append(loc["bounding_box"])
                            labels.append("anomaly")  # Standard label for compatibility
                            scores.append(loc.get("confidence", 1.0))

                    pred["boxes"] = boxes
                    pred["labels"] = labels
                    pred["scores"] = scores

                boxes_len = len(pred.get("boxes", []))
                if "labels" not in pred:
                    pred["labels"] = ["anomaly"] * boxes_len
                if "scores" not in pred:
                    pred["scores"] = [1.0] * boxes_len
            fw.write(json.dumps(pred) + "\n")

    # Create references file
    refs_file = main_run_dir / "refs.jsonl"
    logger.info(f"Saving references to: {refs_file}")
    with open(refs_file, "w") as fr:
        for i, rec in enumerate(huggingface_dataset):
            if i >= len(preds):
                break
            bg = rec.get("bbox_gold", {})
            boxes = [
                [x, y, x + w, y + h]
                for x, y, w, h in zip(
                    bg.get("x", []), bg.get("y", []), bg.get("width", []), bg.get("height", []), strict=False
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
    metrics = evaluate(str(preds_file), str(refs_file), task=config.task)
    logger.info(f"Overall evaluation metrics: {metrics}")

    # Return summary of the run
    return {"run_dir": str(main_run_dir), "metrics": metrics}


def create_model_adapter(config: Config) -> OpenAIAdapter:
    """Set up the model adapter without fallback support."""
    logger.info(f"Setting up adapter for model: {config.model.name}")
    return OpenAIAdapter(
        model_name=config.model.name,
        max_retries=config.model.max_retries,
        timeout=config.model.timeout,
    )


def create_retriever_instance(config: Config) -> BM25Retriever | DenseRetriever | HybridRetriever | None:
    """Set up the retriever based on configuration."""
    # Enable retrieval for retrieval and comprehensive approaches, or if explicitly enabled
    needs_retrieval = (
        config.use_retrieval
        or config.approach in ["retrieval", "comprehensive"]
        or (config.approach == "multiturn" and config.use_retrieval)
    )

    if not needs_retrieval:
        return None

    bm25_idx = Path(to_absolute_path(config.paths.index_dir)) / "bm25"
    faiss_idx = Path(to_absolute_path(config.paths.index_dir)) / "faiss"

    try:
        if config.retrieval.type == "bm25":
            logger.info(f"Setting up BM25Retriever from {bm25_idx}")
            return BM25Retriever(str(bm25_idx))
        elif config.retrieval.type == "dense":
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
                alpha=config.retrieval.hybrid_ratio,
                reranker=reranker,
                query_expander=query_expander,
            )
    except Exception as e:
        logger.error(f"Failed to setup retriever (type: {config.retrieval.type}): {e}")
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


def process_batch_baseline(
    batch_idx: int,
    batch: dict[str, Any],
    main_run_dir: Path,
    task: str,
    config: Config,
    adapter: Any,  # BaseAdapter when properly imported
    retriever: Any | None,  # BaseRetriever when properly imported
    preds: list[dict[str, Any]],
    huggingface_dataset: Any,  # Dataset when properly imported
) -> None:
    """Process a single batch from the dataloader."""
    # ------------------------------------------------------------------
    # Persist the image at its ORIGINAL resolution - critical for proper
    # bounding-box visualisation.  We therefore reload the image directly
    # from the HuggingFace dataset (which retains the full-size image) and
    # bypass any resize transforms that were applied for model ingestion.
    # ------------------------------------------------------------------

    from PIL import Image  # local import to avoid top-level PIL dependency

    hf_record = huggingface_dataset[batch_idx]

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

    metadata_rec = extract_batch_metadata(batch)

    if config.use_retrieval and retriever:
        query = (
            metadata_rec.get("final_diagnosis")
            or metadata_rec.get("diagnosis")
            or metadata_rec.get("caption")
            or metadata_rec.get("clinical_history")
            or ""
        )
        retrieval_debug_info = {"query": query, "success": False, "error": None, "passages": []}
        try:
            passages = retriever(query, k=config.retrieval.top_k) if query else []
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
        if isinstance(_meta_list, list | tuple) and _meta_list and isinstance(_meta_list[0], dict):
            metadata_for_prompt["clinical_history"] = _meta_list[0].get("clinical_history", "")
        else:
            metadata_for_prompt["clinical_history"] = ""

    # Load prompt with enhanced system prompt integration
    prompt = load_jinja_template_prompt(
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

    result = parse_model_json_response(text)

    # Add metadata
    result["image_path"] = str(img_path)
    result["ground_truth_image_idx"] = batch_idx

    # Ensure required keys are present
    ensure_evaluation_keys(result)

    logger.info(f"Successfully processed result for image {batch_idx}")

    # Save individual prediction for this image
    save_prediction(img_folder, result)

    # Generate and save reference
    save_reference(img_folder, batch_idx, huggingface_dataset)

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
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib import patheffects as pe
        from PIL import Image

        img = Image.open(img_path).convert("L")
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(img, cmap="gray")

        def _iter_boxes(raw_boxes):
            for b in raw_boxes:
                if isinstance(b, list | tuple) and len(b) == 4:
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

    gt_bg = huggingface_dataset[batch_idx].get("bbox_gold", {})
    gt_boxes = [
        [x, y, x + w, y + h]
        for x, y, w, h in zip(
            gt_bg.get("x", []), gt_bg.get("y", []), gt_bg.get("width", []), gt_bg.get("height", []), strict=False
        )
    ]
    viz_path = img_folder / "bboxes.png"
    _draw_boxes(img_path, gt_boxes, result.get("boxes", []), viz_path)

    # Evaluate this individual prediction
    evaluate_prediction(img_folder, task)

    # Add to combined predictions list
    preds.append(result)

    # ------------------------------------------------------------------
    # Shared post-processing (ensures keys, saves files, visualises, evals)
    # ------------------------------------------------------------------

    from nova_retrieval_vlm.utils.batch_processing_utils import BatchContext
    from nova_retrieval_vlm.utils.batch_processing_utils import postprocess_batch_result

    ctx = BatchContext(
        idx=batch_idx,
        folder=img_folder,
        img_path=img_path,
        width=pil.width,
        height=pil.height,
    )

    postprocess_batch_result(ctx, result, task, huggingface_dataset, preds)


def ensure_evaluation_keys(result: dict) -> None:
    """Ensure all required keys are present in the result dict."""
    # Convert new localization schema to legacy format if needed
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


def save_prediction(img_folder: Path, result: dict) -> None:
    """Save prediction to file."""
    pred_file = img_folder / "pred.jsonl"
    with open(pred_file, "w") as fw:
        fw.write(json.dumps(result) + "\n")


def save_reference(img_folder: Path, batch_idx: int, huggingface_dataset: Any) -> None:
    """Save reference to file."""
    ref_file = img_folder / "ref.jsonl"
    rec = huggingface_dataset[batch_idx]
    bg = rec.get("bbox_gold", {})
    boxes = [
        [x, y, x + w, y + h]
        for x, y, w, h in zip(
            bg.get("x", []), bg.get("y", []), bg.get("width", []), bg.get("height", []), strict=False
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


def evaluate_prediction(img_folder: Path, task: str) -> dict[str, float]:
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
    config: Config,  # noqa: ARG001 - May be used in future enhancements
    adapter,
    retriever,  # noqa: ARG001 - May be used in future enhancements
    preds: list,
    huggingface_dataset,
):
    """Process a single batch using new chain of thought multi-turn reasoning approach."""
    from PIL import Image  # local import to avoid top-level PIL dependency

    hf_record = huggingface_dataset[batch_idx]

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
        if isinstance(_meta_list, list | tuple) and _meta_list and isinstance(_meta_list[0], dict):
            metadata_for_prompt["clinical_history"] = _meta_list[0].get("clinical_history", "")
        else:
            metadata_for_prompt["clinical_history"] = ""

    # Initialize chain of thought tracking
    chain_of_thought_data = {
        "turns_completed": [],
        "turn_results": {},
        "total_turns": 0,
        "reasoning_complete": False,
        "final_analysis_complete": False,
    }

    # Chain of thought reasoning loop (up to 3 turns)
    max_turns = 3
    current_turn = 0
    previous_turns = []

    while current_turn < max_turns and not chain_of_thought_data["reasoning_complete"]:
        current_turn += 1
        logger.info(f"Multi-turn Turn {current_turn} for image {batch_idx}")

        # Prepare metadata for this turn
        turn_metadata = metadata_for_prompt.copy()
        turn_metadata["turn_number"] = current_turn
        turn_metadata["previous_turns"] = previous_turns
        turn_metadata["approach"] = "multiturn"

        # Load chain of thought prompt
        chain_prompt = load_jinja_template_prompt(
            template_name="multiturn/chain_of_thought.jinja",
            image_path=img_path,
            passages=[],
            metadata=turn_metadata,
        )

        with open(img_folder / f"turn_{current_turn}_prompt.txt", "w") as f:
            f.write(chain_prompt)

        turn_text, turn_log = asyncio.run(
            adapter.generate(img_path, [], system_prompt=chain_prompt)
        )
        logger.info(f"Turn {current_turn} generation cost: {turn_log}")

        with open(img_folder / f"turn_{current_turn}_output.txt", "w") as f:
            f.write(turn_text)

        turn_result = parse_model_json_response(turn_text)
        chain_of_thought_data["turns_completed"].append(f"turn_{current_turn}")
        chain_of_thought_data["turn_results"][f"turn_{current_turn}"] = turn_result
        chain_of_thought_data["total_turns"] += 1

        # Add to previous turns for next iteration
        previous_turns.append(turn_result)

        # Check if reasoning is complete
        reasoning_complete = turn_result.get("reasoning_complete", False)
        continue_reasoning = turn_result.get("continue_reasoning", True)

        if reasoning_complete or not continue_reasoning:
            logger.info(f"Turn {current_turn} indicates reasoning complete")
            chain_of_thought_data["reasoning_complete"] = True
            break

    # Generate final task output
    logger.info(f"Generating final task output for image {batch_idx}")

    # Prepare metadata for final task output
    final_metadata = metadata_for_prompt.copy()
    final_metadata["task"] = task
    final_metadata["chain_of_thought_turns"] = previous_turns
    final_metadata["overall_confidence"] = (
        max([turn.get("confidence", 0.0) for turn in previous_turns]) if previous_turns else 0.0
    )

    # Extract key findings and differential diagnoses from all turns
    all_findings = []
    all_differential_diagnoses = []
    for turn in previous_turns:
        all_findings.extend(turn.get("findings", []))
        all_differential_diagnoses.extend(turn.get("differential_diagnoses", []))

    final_metadata["key_findings"] = list(set(all_findings))  # Remove duplicates
    final_metadata["final_differential_diagnoses"] = list(
        set(all_differential_diagnoses)
    )  # Remove duplicates

    # Load final task output prompt
    final_prompt = load_jinja_template_prompt(
        template_name="multiturn/final_task_output.jinja",
        image_path=img_path,
        passages=[],
        metadata=final_metadata,
    )

    with open(img_folder / "final_task_prompt.txt", "w") as f:
        f.write(final_prompt)

    final_text, final_log = asyncio.run(adapter.generate(img_path, [], system_prompt=final_prompt))
    logger.info(f"Final task output generation cost: {final_log}")

    with open(img_folder / "final_task_output.txt", "w") as f:
        f.write(final_text)

    # Parse final result
    result = parse_model_json_response(final_text)

    # Add metadata
    result["image_path"] = str(img_path)
    result["ground_truth_image_idx"] = batch_idx
    result["chain_of_thought_data"] = chain_of_thought_data

    # Ensure required keys are present
    ensure_evaluation_keys(result)

    logger.info(
        f"Successfully processed multi-turn result for image {batch_idx} ({chain_of_thought_data['total_turns']} turns)"
    )

    # Shared post-processing (ensures keys, saves files, visualises, evals)
    ctx = BatchContext(
        idx=batch_idx,
        folder=img_folder,
        img_path=img_path,
        width=pil.width,
        height=pil.height,
    )

    postprocess_batch_result(ctx, result, task, huggingface_dataset, preds)


def process_batch_visual_multiturn(
    batch_idx: int,
    batch: dict,
    main_run_dir: Path,
    task: str,
    config: Config,  # noqa: ARG001 - May be used in future enhancements
    adapter,
    retriever,  # noqa: ARG001 - May be used in future enhancements
    preds: list,
    huggingface_dataset,
):
    """Process a single batch using new visual chain of thought multi-turn reasoning approach."""
    from PIL import Image  # local import to avoid top-level PIL dependency

    hf_record = huggingface_dataset[batch_idx]

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
        if isinstance(_meta_list, list | tuple) and _meta_list and isinstance(_meta_list[0], dict):
            metadata_for_prompt["clinical_history"] = _meta_list[0].get("clinical_history", "")
        else:
            metadata_for_prompt["clinical_history"] = ""

    # Initialize visual chain of thought tracking
    visual_chain_data = {
        "turns_completed": [],
        "turn_results": {},
        "total_turns": 0,
        "reasoning_complete": False,
        "final_analysis_complete": False,
        "visual_operations_applied": [],
    }

    # Visual chain of thought reasoning loop (up to 3 turns)
    max_turns = 3
    current_turn = 0
    previous_turns = []
    current_image_path = img_path

    while current_turn < max_turns and not visual_chain_data["reasoning_complete"]:
        current_turn += 1
        logger.info(f"Visual multi-turn Turn {current_turn} for image {batch_idx}")

        # Prepare metadata for this turn
        turn_metadata = metadata_for_prompt.copy()
        turn_metadata["turn_number"] = current_turn
        turn_metadata["previous_turns"] = previous_turns
        turn_metadata["approach"] = "visual"

        # Load visual chain of thought prompt
        chain_prompt = load_jinja_template_prompt(
            template_name="multiturn/chain_of_thought.jinja",
            image_path=current_image_path,
            passages=[],
            metadata=turn_metadata,
        )

        with open(img_folder / f"turn_{current_turn}_prompt.txt", "w") as f:
            f.write(chain_prompt)

        turn_text, turn_log = asyncio.run(
            adapter.generate(current_image_path, [], system_prompt=chain_prompt)
        )
        logger.info(f"Turn {current_turn} generation cost: {turn_log}")

        with open(img_folder / f"turn_{current_turn}_output.txt", "w") as f:
            f.write(turn_text)

        turn_result = parse_model_json_response(turn_text)
        visual_chain_data["turns_completed"].append(f"turn_{current_turn}")
        visual_chain_data["turn_results"][f"turn_{current_turn}"] = turn_result
        visual_chain_data["total_turns"] += 1

        # Apply visual operations if requested
        visual_ops = turn_result.get("visual_operations", {})
        if visual_ops.get("requested", False):
            try:
                # Load the current image
                current_image = Image.open(current_image_path).convert("L")

                # Apply zoom if requested
                if visual_ops.get("zoom_factor") and visual_ops["zoom_factor"] != 1.0:
                    current_image = zoom_image(current_image, visual_ops["zoom_factor"])

                # Apply crop if requested
                if visual_ops.get("crop_box"):
                    current_image = crop_image(current_image, visual_ops["crop_box"])

                # Apply contrast if requested
                if visual_ops.get("contrast_factor") and visual_ops["contrast_factor"] != 1.0:
                    current_image = adjust_contrast(current_image, visual_ops["contrast_factor"])

                # Apply intensity threshold if requested
                if visual_ops.get("intensity_range"):
                    current_image = apply_intensity_threshold(
                        current_image,
                        visual_ops["intensity_range"][0],
                        visual_ops["intensity_range"][1],
                    )

                # Save the modified image
                modified_image_path = img_folder / f"modified_round_{current_turn}.png"
                current_image.save(modified_image_path)
                current_image_path = modified_image_path

                # Record the visual operation
                visual_chain_data["visual_operations_applied"].append(
                    {
                        "turn": current_turn,
                        "operations": visual_ops,
                        "modified_image_path": str(modified_image_path),
                    }
                )

                logger.info(f"Applied visual operations for turn {current_turn}")

            except Exception as exc:
                logger.warning(f"Visual operations failed for turn {current_turn}: {exc}")
                # Continue with original image if operations fail
                current_image_path = img_path

        # Add to previous turns for next iteration
        previous_turns.append(turn_result)

        # Check if reasoning is complete
        reasoning_complete = turn_result.get("reasoning_complete", False)
        continue_reasoning = turn_result.get("continue_reasoning", True)

        if reasoning_complete or not continue_reasoning:
            logger.info(f"Turn {current_turn} indicates reasoning complete")
            visual_chain_data["reasoning_complete"] = True
            break

    # Generate final task output
    logger.info(f"Generating final task output for image {batch_idx}")

    # Prepare metadata for final task output
    final_metadata = metadata_for_prompt.copy()
    final_metadata["task"] = task
    final_metadata["chain_of_thought_turns"] = previous_turns
    final_metadata["overall_confidence"] = (
        max([turn.get("confidence", 0.0) for turn in previous_turns]) if previous_turns else 0.0
    )

    # Extract key findings and differential diagnoses from all turns
    all_findings = []
    all_differential_diagnoses = []
    for turn in previous_turns:
        all_findings.extend(turn.get("findings", []))
        all_differential_diagnoses.extend(turn.get("differential_diagnoses", []))

    final_metadata["key_findings"] = list(set(all_findings))  # Remove duplicates
    final_metadata["final_differential_diagnoses"] = list(
        set(all_differential_diagnoses)
    )  # Remove duplicates

    # Load final task output prompt
    final_prompt = load_jinja_template_prompt(
        template_name="multiturn/final_task_output.jinja",
        image_path=img_path,  # Use original image for final output
        passages=[],
        metadata=final_metadata,
    )

    with open(img_folder / "final_task_prompt.txt", "w") as f:
        f.write(final_prompt)

    final_text, final_log = asyncio.run(adapter.generate(img_path, [], system_prompt=final_prompt))
    logger.info(f"Final task output generation cost: {final_log}")

    with open(img_folder / "final_task_output.txt", "w") as f:
        f.write(final_text)

    # Parse final result
    result = parse_model_json_response(final_text)

    # Add metadata
    result["image_path"] = str(img_path)
    result["ground_truth_image_idx"] = batch_idx
    result["visual_chain_data"] = visual_chain_data

    # Ensure required keys are present
    ensure_evaluation_keys(result)

    logger.info(
        f"Successfully processed visual multi-turn result for image {batch_idx} ({visual_chain_data['total_turns']} turns)"
    )

    # Shared post-processing (ensures keys, saves files, visualises, evals)
    ctx = BatchContext(
        idx=batch_idx,
        folder=img_folder,
        img_path=img_path,
        width=pil.width,
        height=pil.height,
    )

    postprocess_batch_result(ctx, result, task, huggingface_dataset, preds)


def process_batch_retrieval(
    batch_idx: int,
    batch: dict,
    main_run_dir: Path,
    task: str,
    config: Config,
    adapter,
    retriever,
    preds: list,
    huggingface_dataset,
):
    """Process a single batch using retrieval-augmented approach."""

    # Get image from HuggingFace dataset (same as baseline)
    from PIL import Image

    hf_record = huggingface_dataset[batch_idx]

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

    logger.info(f"Processing retrieval-augmented analysis for image {batch_idx}: {img_path}")

    # Setup metadata for prompt
    metadata_for_prompt = {
        "image_id": batch_idx,
        "width": pil.width,
        "height": pil.height,
    }

    # Add clinical history for diagnosis task
    if task == "diagnosis":
        _meta_list = batch.get("metadata", []) or []
        if isinstance(_meta_list, list | tuple) and _meta_list and isinstance(_meta_list[0], dict):
            metadata_for_prompt["clinical_history"] = _meta_list[0].get("clinical_history", "")
        else:
            metadata_for_prompt["clinical_history"] = ""

    # Perform retrieval if retriever is available
    passages = []
    retrieval_debug_info = None

    if config.use_retrieval and retriever:
        # Use metadata to create query (same as baseline)
        _meta_list = batch.get("metadata", []) or []
        if isinstance(_meta_list, list | tuple) and _meta_list and isinstance(_meta_list[0], dict):
            metadata_rec = _meta_list[0]
        else:
            metadata_rec = {}

        query = (
            metadata_rec.get("final_diagnosis")
            or metadata_rec.get("diagnosis")
            or metadata_rec.get("caption")
            or metadata_rec.get("clinical_history")
            or ""
        )

        retrieval_debug_info = {"query": query, "success": False, "error": None, "passages": []}

        try:
            passages = retriever(query, k=config.retrieval.top_k) if query else []
            retrieval_debug_info["success"] = True
            retrieval_debug_info["passages"] = passages
            logger.info(f"Retrieved {len(passages)} passages for image {batch_idx}")
        except Exception as exc:
            logger.warning(f"Retrieval failed for image {batch_idx}: {exc}")
            retrieval_debug_info["error"] = str(exc)

    # Save retrieval debug info
    if retrieval_debug_info:
        with open(img_folder / "retrieval_debug.txt", "w") as f:
            f.write(f"Query: {retrieval_debug_info['query']}\n")
            f.write(f"Success: {retrieval_debug_info['success']}\n")
            if retrieval_debug_info.get("error"):
                f.write(f"Error: {retrieval_debug_info['error']}\n")
            f.write("\n=== PASSAGES ===\n")
            for p in passages:
                f.write(p + "\n\n")

    # Select template based on retrieval success (same as baseline)
    template_name = f"retrieval_{task}.jinja" if passages else f"baseline/{task}.jinja"

    # Load enhanced prompt
    prompt = load_jinja_template_prompt(
        template_name=template_name,
        image_path=img_path,
        passages=passages,
        metadata=metadata_for_prompt,
    )

    with open(img_folder / "prompt.txt", "w") as f:
        f.write(prompt)

    # Generate response
    text, log = asyncio.run(adapter.generate(img_path, passages, system_prompt=prompt))
    logger.info(f"Retrieval-augmented generation cost: {log}")

    # Save raw output for debugging
    with open(img_folder / "raw_output.txt", "w") as f:
        f.write(text)

    # Parse result
    result = parse_model_json_response(text)

    # Add metadata
    result["image_path"] = str(img_path)
    result["ground_truth_image_idx"] = batch_idx

    # Ensure required keys are present
    ensure_evaluation_keys(result)

    logger.info(f"Successfully processed result for image {batch_idx}")

    # Save individual prediction for this image
    save_prediction(img_folder, result)

    # Generate and save reference
    save_reference(img_folder, batch_idx, huggingface_dataset)

    # Shared post-processing (ensures keys, saves files, visualises, evals)
    from nova_retrieval_vlm.utils.batch_processing_utils import BatchContext
    from nova_retrieval_vlm.utils.batch_processing_utils import postprocess_batch_result

    ctx = BatchContext(
        idx=batch_idx,
        folder=img_folder,
        img_path=img_path,
        width=pil.width,
        height=pil.height,
    )

    postprocess_batch_result(ctx, result, task, huggingface_dataset, preds)


def process_batch_web_search(
    batch_idx: int,
    batch: dict,
    main_run_dir: Path,
    task: str,
    config: Config,
    adapter,
    retriever,
    preds: list,
    huggingface_dataset,
):
    """Process a single batch using new web search chain of thought multi-turn reasoning approach."""
    from PIL import Image  # local import to avoid top-level PIL dependency

    hf_record = huggingface_dataset[batch_idx]

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
        if isinstance(_meta_list, list | tuple) and _meta_list and isinstance(_meta_list[0], dict):
            metadata_for_prompt["clinical_history"] = _meta_list[0].get("clinical_history", "")
        else:
            metadata_for_prompt["clinical_history"] = ""

    # Initialize web search chain of thought tracking
    web_chain_data = {
        "turns_completed": [],
        "turn_results": {},
        "total_turns": 0,
        "reasoning_complete": False,
        "final_analysis_complete": False,
        "web_searches_performed": [],
    }

    # Initialize web searcher
    web_searcher = MedicalWebSearcher() if config.use_web_search else None

    # Web search chain of thought reasoning loop (up to 3 turns)
    max_turns = 3
    current_turn = 0
    previous_turns = []
    web_search_results = []

    while current_turn < max_turns and not web_chain_data["reasoning_complete"]:
        current_turn += 1
        logger.info(f"Web search multi-turn Turn {current_turn} for image {batch_idx}")

        # Prepare metadata for this turn
        turn_metadata = metadata_for_prompt.copy()
        turn_metadata["turn_number"] = current_turn
        turn_metadata["previous_turns"] = previous_turns
        turn_metadata["approach"] = "web_search"
        turn_metadata["web_search_results"] = web_search_results

        # Load web search chain of thought prompt
        chain_prompt = load_jinja_template_prompt(
            template_name="multiturn/chain_of_thought.jinja",
            image_path=img_path,
            passages=[],
            metadata=turn_metadata,
            mode="web_search",
        )

        with open(img_folder / f"turn_{current_turn}_prompt.txt", "w") as f:
            f.write(chain_prompt)

        turn_text, turn_log = asyncio.run(
            adapter.generate(img_path, [], system_prompt=chain_prompt)
        )
        logger.info(f"Turn {current_turn} generation cost: {turn_log}")

        with open(img_folder / f"turn_{current_turn}_output.txt", "w") as f:
            f.write(turn_text)

        turn_result = parse_model_json_response(turn_text)
        web_chain_data["turns_completed"].append(f"turn_{current_turn}")
        web_chain_data["turn_results"][f"turn_{current_turn}"] = turn_result
        web_chain_data["total_turns"] += 1

        # Perform web search if requested
        web_search = turn_result.get("web_search", {})
        if web_search.get("requested", False) and web_searcher:
            try:
                search_type = web_search.get("search_type", "diagnosis")
                query = web_search.get("query", "")

                if search_type == "diagnosis":
                    results = web_searcher.medical_search(query)
                elif search_type == "guidelines":
                    results = web_searcher.guidelines_search(query)
                elif search_type == "research":
                    results = web_searcher.research_search(query)
                elif search_type == "anatomy":
                    results = web_searcher.anatomy_search(query)
                else:
                    results = web_searcher.general_search(query)

                web_search_results.append(f"Query: {query}\nResults: {results}")

                # Record the web search (convert WebSearchResult objects to dicts for JSON serialization)
                results_dicts = []
                for result in results:
                    results_dicts.append(
                        {
                            "title": result.title,
                            "url": result.url,
                            "snippet": result.snippet,
                            "source": result.source,
                            "relevance_score": result.relevance_score,
                            "medical_concepts": result.medical_concepts,
                            "publication_date": result.publication_date,
                        }
                    )

                web_chain_data["web_searches_performed"].append(
                    {
                        "turn": current_turn,
                        "search_type": search_type,
                        "query": query,
                        "results": results_dicts,
                    }
                )

                logger.info(
                    f"Performed web search for turn {current_turn}: {search_type} - {query}"
                )

            except Exception as exc:
                logger.warning(f"Web search failed for turn {current_turn}: {exc}")
                web_search_results.append(f"Query: {query}\nError: {str(exc)}")

        # Add to previous turns for next iteration
        previous_turns.append(turn_result)

        # Check if reasoning is complete
        reasoning_complete = turn_result.get("reasoning_complete", False)
        continue_reasoning = turn_result.get("continue_reasoning", True)

        if reasoning_complete or not continue_reasoning:
            logger.info(f"Turn {current_turn} indicates reasoning complete")
            web_chain_data["reasoning_complete"] = True
            break

    # Generate final task output
    logger.info(f"Generating final task output for image {batch_idx}")

    # Prepare metadata for final task output
    final_metadata = metadata_for_prompt.copy()
    final_metadata["task"] = task
    final_metadata["chain_of_thought_turns"] = previous_turns
    final_metadata["overall_confidence"] = (
        max([turn.get("confidence", 0.0) for turn in previous_turns]) if previous_turns else 0.0
    )

    # Extract key findings and differential diagnoses from all turns
    all_findings = []
    all_differential_diagnoses = []
    for turn in previous_turns:
        all_findings.extend(turn.get("findings", []))
        all_differential_diagnoses.extend(turn.get("differential_diagnoses", []))

    final_metadata["key_findings"] = list(set(all_findings))  # Remove duplicates
    final_metadata["final_differential_diagnoses"] = list(
        set(all_differential_diagnoses)
    )  # Remove duplicates

    # Load final task output prompt
    final_prompt = load_jinja_template_prompt(
        template_name="multiturn/final_task_output.jinja",
        image_path=img_path,
        passages=[],
        metadata=final_metadata,
        mode="web_search",
    )

    with open(img_folder / "final_task_prompt.txt", "w") as f:
        f.write(final_prompt)

    final_text, final_log = asyncio.run(adapter.generate(img_path, [], system_prompt=final_prompt))
    logger.info(f"Final task output generation cost: {final_log}")

    with open(img_folder / "final_task_output.txt", "w") as f:
        f.write(final_text)

    # Parse final result
    result = parse_model_json_response(final_text)

    # Add metadata
    result["image_path"] = str(img_path)
    result["ground_truth_image_idx"] = batch_idx
    result["web_chain_data"] = web_chain_data

    # Ensure required keys are present
    ensure_evaluation_keys(result)

    logger.info(
        f"Successfully processed web search multi-turn result for image {batch_idx} ({web_chain_data['total_turns']} turns)"
    )

    # Shared post-processing (ensures keys, saves files, visualises, evals)
    ctx = BatchContext(
        idx=batch_idx,
        folder=img_folder,
        img_path=img_path,
        width=pil.width,
        height=pil.height,
    )

    postprocess_batch_result(ctx, result, task, huggingface_dataset, preds)


def process_batch_comprehensive(
    batch_idx: int,
    batch: dict,
    main_run_dir: Path,
    task: str,
    config: Config,
    adapter,
    retriever,
    preds: list,
    huggingface_dataset,
):
    """Process a single batch using new comprehensive chain of thought multi-turn reasoning approach."""
    from PIL import Image  # local import to avoid top-level PIL dependency

    hf_record = huggingface_dataset[batch_idx]

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
        if isinstance(_meta_list, list | tuple) and _meta_list and isinstance(_meta_list[0], dict):
            metadata_for_prompt["clinical_history"] = _meta_list[0].get("clinical_history", "")
        else:
            metadata_for_prompt["clinical_history"] = ""

    # Initialize comprehensive chain of thought tracking
    comprehensive_chain_data = {
        "turns_completed": [],
        "turn_results": {},
        "total_turns": 0,
        "reasoning_complete": False,
        "final_analysis_complete": False,
        "visual_operations_applied": [],
        "web_searches_performed": [],
    }

    # Initialize web searcher
    web_searcher = MedicalWebSearcher() if config.use_web_search else None

    # Comprehensive chain of thought reasoning loop (up to 3 turns)
    max_turns = 3
    current_turn = 0
    previous_turns = []
    current_image_path = img_path
    web_search_results = []

    while current_turn < max_turns and not comprehensive_chain_data["reasoning_complete"]:
        current_turn += 1
        logger.info(f"Comprehensive multi-turn Turn {current_turn} for image {batch_idx}")

        # Prepare metadata for this turn
        turn_metadata = metadata_for_prompt.copy()
        turn_metadata["turn_number"] = current_turn
        turn_metadata["previous_turns"] = previous_turns
        turn_metadata["approach"] = "comprehensive"
        turn_metadata["web_search_results"] = web_search_results

        # Load comprehensive chain of thought prompt
        chain_prompt = load_jinja_template_prompt(
            template_name="multiturn/chain_of_thought.jinja",
            image_path=current_image_path,
            passages=[],
            metadata=turn_metadata,
            mode="comprehensive",
        )

        with open(img_folder / f"turn_{current_turn}_prompt.txt", "w") as f:
            f.write(chain_prompt)

        turn_text, turn_log = asyncio.run(
            adapter.generate(current_image_path, [], system_prompt=chain_prompt)
        )
        logger.info(f"Turn {current_turn} generation cost: {turn_log}")

        with open(img_folder / f"turn_{current_turn}_output.txt", "w") as f:
            f.write(turn_text)

        turn_result = parse_model_json_response(turn_text)
        comprehensive_chain_data["turns_completed"].append(f"turn_{current_turn}")
        comprehensive_chain_data["turn_results"][f"turn_{current_turn}"] = turn_result
        comprehensive_chain_data["total_turns"] += 1

        # Apply visual operations if requested
        visual_ops = turn_result.get("visual_operations", {})
        if visual_ops.get("requested", False):
            try:
                # Load the current image
                current_image = Image.open(current_image_path).convert("L")

                # Apply zoom if requested
                if visual_ops.get("zoom_factor") and visual_ops["zoom_factor"] != 1.0:
                    current_image = zoom_image(current_image, visual_ops["zoom_factor"])

                # Apply crop if requested
                if visual_ops.get("crop_box"):
                    current_image = crop_image(current_image, visual_ops["crop_box"])

                # Apply contrast if requested
                if visual_ops.get("contrast_factor") and visual_ops["contrast_factor"] != 1.0:
                    current_image = adjust_contrast(current_image, visual_ops["contrast_factor"])

                # Apply intensity threshold if requested
                if visual_ops.get("intensity_range"):
                    current_image = apply_intensity_threshold(
                        current_image,
                        visual_ops["intensity_range"][0],
                        visual_ops["intensity_range"][1],
                    )

                # Save the modified image
                modified_image_path = img_folder / f"modified_round_{current_turn}.png"
                current_image.save(modified_image_path)
                current_image_path = modified_image_path

                # Record the visual operation
                comprehensive_chain_data["visual_operations_applied"].append(
                    {
                        "turn": current_turn,
                        "operations": visual_ops,
                        "modified_image_path": str(modified_image_path),
                    }
                )

                logger.info(f"Applied visual operations for turn {current_turn}")

            except Exception as exc:
                logger.warning(f"Visual operations failed for turn {current_turn}: {exc}")
                # Continue with original image if operations fail
                current_image_path = img_path

        # Perform web search if requested
        web_search = turn_result.get("web_search", {})
        if web_search.get("requested", False) and web_searcher:
            try:
                search_type = web_search.get("search_type", "diagnosis")
                query = web_search.get("query", "")

                if search_type == "diagnosis":
                    results = web_searcher.medical_search(query)
                elif search_type == "guidelines":
                    results = web_searcher.guidelines_search(query)
                elif search_type == "research":
                    results = web_searcher.research_search(query)
                elif search_type == "anatomy":
                    results = web_searcher.anatomy_search(query)
                else:
                    results = web_searcher.general_search(query)

                web_search_results.append(f"Query: {query}\nResults: {results}")

                # Record the web search (convert WebSearchResult objects to dicts for JSON serialization)
                results_dicts = []
                for result in results:
                    results_dicts.append(
                        {
                            "title": result.title,
                            "url": result.url,
                            "snippet": result.snippet,
                            "source": result.source,
                            "relevance_score": result.relevance_score,
                            "medical_concepts": result.medical_concepts,
                            "publication_date": result.publication_date,
                        }
                    )

                comprehensive_chain_data["web_searches_performed"].append(
                    {
                        "turn": current_turn,
                        "search_type": search_type,
                        "query": query,
                        "results": results_dicts,
                    }
                )

                logger.info(
                    f"Performed web search for turn {current_turn}: {search_type} - {query}"
                )

            except Exception as exc:
                logger.warning(f"Web search failed for turn {current_turn}: {exc}")
                web_search_results.append(f"Query: {query}\nError: {str(exc)}")

        # Add to previous turns for next iteration
        previous_turns.append(turn_result)

        # Check if reasoning is complete
        reasoning_complete = turn_result.get("reasoning_complete", False)
        continue_reasoning = turn_result.get("continue_reasoning", True)

        if reasoning_complete or not continue_reasoning:
            logger.info(f"Turn {current_turn} indicates reasoning complete")
            comprehensive_chain_data["reasoning_complete"] = True
            break

    # Generate final task output
    logger.info(f"Generating final task output for image {batch_idx}")

    # Prepare metadata for final task output
    final_metadata = metadata_for_prompt.copy()
    final_metadata["task"] = task
    final_metadata["chain_of_thought_turns"] = previous_turns
    final_metadata["overall_confidence"] = (
        max([turn.get("confidence", 0.0) for turn in previous_turns]) if previous_turns else 0.0
    )

    # Extract key findings and differential diagnoses from all turns
    all_findings = []
    all_differential_diagnoses = []
    for turn in previous_turns:
        all_findings.extend(turn.get("findings", []))
        all_differential_diagnoses.extend(turn.get("differential_diagnoses", []))

    final_metadata["key_findings"] = list(set(all_findings))  # Remove duplicates
    final_metadata["final_differential_diagnoses"] = list(
        set(all_differential_diagnoses)
    )  # Remove duplicates

    # Load final task output prompt
    final_prompt = load_jinja_template_prompt(
        template_name="multiturn/final_task_output.jinja",
        image_path=img_path,  # Use original image for final output
        passages=[],
        metadata=final_metadata,
        mode="comprehensive",
    )

    with open(img_folder / "final_task_prompt.txt", "w") as f:
        f.write(final_prompt)

    final_text, final_log = asyncio.run(adapter.generate(img_path, [], system_prompt=final_prompt))
    logger.info(f"Final task output generation cost: {final_log}")

    with open(img_folder / "final_task_output.txt", "w") as f:
        f.write(final_text)

    # Parse final result
    result = parse_model_json_response(final_text)

    # Add metadata
    result["image_path"] = str(img_path)
    result["ground_truth_image_idx"] = batch_idx
    result["comprehensive_chain_data"] = comprehensive_chain_data

    # Ensure required keys are present
    ensure_evaluation_keys(result)

    logger.info(
        f"Successfully processed comprehensive multi-turn result for image {batch_idx} ({comprehensive_chain_data['total_turns']} turns)"
    )

    # Shared post-processing (ensures keys, saves files, visualises, evals)
    ctx = BatchContext(
        idx=batch_idx,
        folder=img_folder,
        img_path=img_path,
        width=pil.width,
        height=pil.height,
    )

    postprocess_batch_result(ctx, result, task, huggingface_dataset, preds)


# ---------------------------------------------------------------------------
# Helper for resumable runs – determines whether a given sample has already
# been processed in a previous (possibly interrupted) session.
# ---------------------------------------------------------------------------


def _skip_if_existing(batch_idx: int, main_run_dir: Path, config: Config) -> bool:  # noqa: D401
    """Return True if *image_{batch_idx}/pred.jsonl* exists in *main_run_dir*.

    The helper honours the **skip_existing** flag in the runtime *config*.
    """

    if not getattr(config, "skip_existing", False):
        return False

    pred_path = main_run_dir / f"image_{batch_idx}" / "pred.jsonl"
    return pred_path.exists()


if __name__ == "__main__":
    # Simple entry point that bypasses Hydra issues
    if len(sys.argv) > 1 and any("=" in arg for arg in sys.argv[1:]):
        # Parse arguments manually and call main directly
        config = _parse_cli_args()
        main(config)
    else:
        # Use Hydra for help and other cases
        main()
