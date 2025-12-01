"""Modern CLI interface using processor pattern."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import hydra
from beartype import beartype
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from PIL import Image

if TYPE_CHECKING:
    pass

# Add project root to path so config module can be imported (required for Hydra)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nova_retrieval_vlm.agentic import AgenticLocalizationProcessor  # noqa: E402
from nova_retrieval_vlm.config import AgenticConfig  # noqa: E402
from nova_retrieval_vlm.config import Config  # noqa: E402
from nova_retrieval_vlm.config import ModelConfig  # noqa: E402
from nova_retrieval_vlm.config import PathsConfig  # noqa: E402
from nova_retrieval_vlm.config import VisualizationConfig  # noqa: E402
from nova_retrieval_vlm.processors import BaseProcessor  # noqa: E402
from nova_retrieval_vlm.processors import LocalizationProcessor  # noqa: E402
from nova_retrieval_vlm.processors import ProcessorConfig  # noqa: E402
from nova_retrieval_vlm.types import BatchData  # noqa: E402
from nova_retrieval_vlm.types import ModelResponse  # noqa: E402


@beartype
def save_per_subject_results(
    batch_idx: int,
    responses: list[ModelResponse | dict[str, Any]],
    config: Config,
    output_dir: Path,
) -> None:
    """Save per-subject results directly from batch responses."""
    # Create per_subject directory if it doesn't exist
    per_subject_dir = output_dir / "per_subject"
    per_subject_dir.mkdir(parents=True, exist_ok=True)

    for i, response in enumerate(responses):
        subject_idx = batch_idx * len(responses) + i
        subject_dir = per_subject_dir / f"subject_{subject_idx:04d}"

        try:
            subject_dir.mkdir(exist_ok=True)

            # Convert ModelResponse to dict if needed
            if isinstance(response, ModelResponse):
                response_dict = response.model_dump()
            else:
                response_dict = response

            # Parse the structured response
            try:
                parsed_response = json.loads(response_dict.get("text", "{}"))
            except json.JSONDecodeError:
                parsed_response = {}

            # Create structured subject data
            subject_data = {
                "subject_id": subject_idx,
                "model": config.model.name,
                "task": "all_tasks",
                "caption": parsed_response.get("caption", {}),
                "diagnosis": parsed_response.get("diagnosis", {}),
                "localization": parsed_response.get("localization", {}),
                "raw_response": response_dict.get("text", ""),
                "confidence": response_dict.get("confidence", 0.0),
                "reasoning": response_dict.get("reasoning", ""),
                "metadata": response_dict.get("metadata", {}),
                "processing_info": {
                    "model": config.model.name,
                    "agentic_enabled": config.agentic.enabled,
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "has_caption": bool(parsed_response.get("caption")),
                    "has_diagnosis": bool(parsed_response.get("diagnosis")),
                    "has_localization": bool(
                        parsed_response.get("localization", {}).get("localizations")
                    ),
                },
            }

            # Save full predictions
            predictions_file = subject_dir / "predictions.json"
            with open(predictions_file, "w") as f:
                json.dump(subject_data, f, indent=2)

            # Save quick summary
            summary_file = subject_dir / "summary.json"
            summary = {
                "subject_id": subject_idx,
                "caption": subject_data["caption"],
                "diagnosis": subject_data["diagnosis"],
                "localization": subject_data["localization"],
                "metadata": subject_data["metadata"],
                "confidence": subject_data["confidence"],
                "processing_info": subject_data["processing_info"],
            }
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Saved results for subject {subject_idx:04d} to {subject_dir}")

        except Exception as e:
            logger.error(f"Error saving subject {subject_idx}: {e}")


def _image_to_path(image: Image.Image | str) -> str:
    """Convert PIL Image to temporary file path, or return existing path."""
    if isinstance(image, str):
        # Already a path
        return image
    else:
        # Save PIL Image to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file.name, format="JPEG", quality=95)
            return temp_file.name


# Unified processors - both handle all tasks (captioning, diagnosis, localization) in one pass
STANDARD_PROCESSOR = LocalizationProcessor  # Handles all tasks via unified prompts
AGENTIC_PROCESSOR = AgenticLocalizationProcessor  # Handles all tasks via unified agentic prompts


@beartype
def create_processor(config: Config) -> BaseProcessor:
    """Create processor instance based on agentic config."""

    # Use unified processors that handle all tasks (captioning, diagnosis, localization) in one pass
    if config.agentic.enabled:
        logger.info("Using agentic processor for all tasks (captioning, diagnosis, localization)")
        processor_cls = AGENTIC_PROCESSOR

        processor_config = ProcessorConfig(
            task_name="all_tasks",  # Unified processor handles all tasks
            model_name=config.model.name,
            batch_size=config.batch_size,
            output_dir=Path(config.paths.output_dir),
            skip_existing=config.skip_existing,
            reasoning_enabled=config.agentic.reasoning_enabled,  # Pass reasoning flag even in single-turn mode
        )

        return processor_cls(
            config=processor_config,
            use_tools=config.agentic.use_tools,
            max_turns=config.agentic.max_turns,
        )

    # Standard processor
    logger.info("Using standard processor for all tasks (captioning, diagnosis, localization)")
    processor_cls = STANDARD_PROCESSOR

    processor_config = ProcessorConfig(
        task_name="all_tasks",  # Unified processor handles all tasks
        model_name=config.model.name,
        batch_size=config.batch_size,
        output_dir=Path(config.paths.output_dir),
        skip_existing=config.skip_existing,
        reasoning_enabled=config.agentic.reasoning_enabled,  # Pass reasoning flag even in single-turn mode
    )

    return processor_cls(processor_config)


@beartype
async def run_task(config: Config) -> dict[str, Any]:
    """Run unified multi-task analysis (captioning, diagnosis, localization) with modern architecture."""
    logger.info("Starting unified NOVA analysis: captioning, diagnosis, localization")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Processing mode: {'agentic' if config.agentic.enabled else 'single-turn'}")
    logger.info(
        f"Tools: {'enabled' if config.agentic.enabled and config.agentic.use_tools else 'disabled'}"
    )
    if config.agentic.enabled:
        logger.info(
            f"Agentic settings: reasoning_enabled={config.agentic.reasoning_enabled}, "
            f"max_turns={config.agentic.max_turns}"
        )

    # Create processor
    processor = create_processor(config)

    # Get dataset directly to avoid PIL Image collation issues
    from nova_retrieval_vlm.data.nova_dataset import NovaDataset

    dataset = NovaDataset(data_dir=config.paths.data_dir, transform=None)
    logger.info(f"Loaded dataset with {len(dataset)} samples")

    # Create simple batches without collation issues
    def simple_batches(dataset, batch_size):
        """Simple batching that handles PIL Images correctly."""
        for i in range(0, len(dataset), batch_size):
            yield [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]

    # Process batches
    all_responses = []
    all_ground_truth = []

    total_batches = (len(dataset) + config.batch_size - 1) // config.batch_size
    max_batches = config.max_iterations if config.max_iterations > 0 else total_batches
    logger.info(
        f"Starting to process {min(max_batches, total_batches)} batches (limit: {max_batches})"
    )

    for batch_idx, batch_items in enumerate(simple_batches(dataset, config.batch_size)):
        # Check max_iterations limit
        if config.max_iterations > 0 and batch_idx >= config.max_iterations:
            logger.info(f"Stopping at max_iterations: {config.max_iterations} batches")
            break
        logger.info(
            f"Processing batch {batch_idx + 1}/{total_batches} (items {batch_idx * config.batch_size + 1}-{min((batch_idx + 1) * config.batch_size, len(dataset))})"
        )

        # batch_items is already a list of dataset items
        logger.debug(f"Batch {batch_idx} has {len(batch_items)} items")

        # Create BatchData from raw batch - fail fast on errors
        batch_data = BatchData(
            images=[
                _image_to_path(item["image"]) if "image" in item else "" for item in batch_items
            ],
            metadata=[item.get("metadata", {}) for item in batch_items],
            labels=(
                [item.get("label") for item in batch_items]
                if batch_items and "label" in batch_items[0]
                else None
            ),
        )

        # Skip if already processed
        if hasattr(processor, "should_skip_batch") and processor.should_skip_batch(batch_idx):
            logger.info(f"Skipping batch {batch_idx} (already processed)")
            responses = processor.load_batch_results(batch_idx)
        else:
            # Process batch - fail fast on any errors
            responses = await processor.process_batch(batch_data, batch_idx)

        # Save per-subject results (for both new and skipped batches)
        output_dir = Path(config.paths.output_dir)
        save_per_subject_results(batch_idx, responses, config, output_dir)

        # Collect results
        if isinstance(responses, list):
            all_responses.extend(responses)
        elif responses is not None:
            all_responses.append(responses)

        if batch_data and batch_data.labels:
            all_ground_truth.extend(batch_data.labels)

    # Evaluate results - optional if ground truth available
    if all_responses and all_ground_truth:
        logger.info("Evaluating results...")
        metrics = processor.evaluate_responses(all_responses, all_ground_truth)
        logger.info(f"Final metrics: {metrics}")
    elif all_responses:
        logger.info(f"Generated {len(all_responses)} responses without ground truth evaluation")
        # Create basic metrics without ground truth comparison
        metrics = {
            "total_responses": len(all_responses),
            "model_name": config.model.name,
            "processing_mode": "single_turn",
            "evaluation": "skipped_no_ground_truth",
        }
    else:
        raise ValueError("No valid responses generated")

    logger.success("Unified NOVA analysis completed successfully!")

    # Convert metrics to dict for consistent return type - fail fast on unexpected types
    if hasattr(metrics, "__dict__"):
        return metrics.__dict__
    elif isinstance(metrics, dict):
        return metrics
    else:
        raise TypeError(f"Unexpected metrics type: {type(metrics)}")


# Use relative path from this file to config directory
# From src/nova_retrieval_vlm/cli.py to config/ is ../../config
@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    try:
        # Convert OmegaConf to Config dataclass with proper nested conversion
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Handle nested conversion for dataclass fields
        if "model" in config_dict:
            config_dict["model"] = ModelConfig(**config_dict["model"])
        if "paths" in config_dict:
            config_dict["paths"] = PathsConfig(**config_dict["paths"])
        if "agentic" in config_dict:
            config_dict["agentic"] = AgenticConfig(**config_dict["agentic"])
        if "visualization" in config_dict:
            config_dict["visualization"] = VisualizationConfig(**config_dict["visualization"])

        config = Config(**config_dict)

        asyncio.run(run_task(config))
    except KeyboardInterrupt:
        logger.info("Task interrupted by user")
    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise


if __name__ == "__main__":
    # Register config
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)

    main()
