"""Modern CLI interface using processor pattern."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from beartype import beartype
from hydra.core.config_store import ConfigStore
from loguru import logger

if TYPE_CHECKING:
    pass

from nova_retrieval_vlm.config import Config
from nova_retrieval_vlm.data.nova_dataset import get_dataloader
from nova_retrieval_vlm.processors import BaseProcessor
from nova_retrieval_vlm.processors import CaptionProcessor
from nova_retrieval_vlm.processors import DetectionProcessor
from nova_retrieval_vlm.processors import DiagnosisProcessor
from nova_retrieval_vlm.processors import LocalizationProcessor
from nova_retrieval_vlm.processors import ProcessorConfig
from nova_retrieval_vlm.types import BatchData

# Register processors
PROCESSORS: dict[str, type[BaseProcessor]] = {
    "localization": LocalizationProcessor,
    "caption": CaptionProcessor,
    "detection": DetectionProcessor,
    "diagnosis": DiagnosisProcessor,
}


@beartype
def create_processor(config: Config) -> BaseProcessor:
    """Create processor instance based on task."""
    if config.task not in PROCESSORS:
        raise ValueError(f"Unknown task: {config.task}. Available: {list(PROCESSORS.keys())}")

    processor_cls = PROCESSORS[config.task]

    processor_config = ProcessorConfig(
        task_name=config.task,
        model_name=config.model.name,
        batch_size=config.batch_size,
        use_retrieval=config.use_retrieval,
        retrieval_type=config.retrieval.type,
        output_dir=Path(config.output_dir),
        skip_existing=config.skip_existing,
    )

    return processor_cls(processor_config)


@beartype
async def run_task(config: Config) -> None:
    """Run the specified task with modern architecture."""
    logger.info(f"Starting task: {config.task}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Use retrieval: {config.use_retrieval}")

    # Create processor
    processor = create_processor(config)

    # Get data loader
    dataloader = get_dataloader(batch_size=config.batch_size, data_dir=config.paths.data_dir)

    # Process batches
    all_responses = []
    all_ground_truth = []

    for batch_idx, batch in enumerate(dataloader):
        logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

        # Convert batch to proper format
        if isinstance(batch, dict):
            # Handle single item batch
            batch_items = [batch]
        elif isinstance(batch, list):
            batch_items = batch
        else:
            logger.warning(f"Unexpected batch type: {type(batch)}")
            continue

        # Skip if already processed
        batch_data = None
        if hasattr(processor, "should_skip_batch") and processor.should_skip_batch(batch_idx):
            logger.info(f"Skipping batch {batch_idx} (already processed)")
            if hasattr(processor, "load_batch_results"):
                responses = processor.load_batch_results(batch_idx)
            else:
                continue
        else:
            # Create BatchData from raw batch
            try:
                batch_data = BatchData(
                    images=[str(item["image"]) if "image" in item else "" for item in batch_items],
                    metadata=[item.get("metadata", {}) for item in batch_items],
                    labels=(
                        [item.get("label") for item in batch_items]
                        if batch_items and "label" in batch_items[0]
                        else None
                    ),
                )

                # Process batch
                try:
                    responses = await processor.process_batch(batch_data, batch_idx)

                    # Save results if processor supports it
                    if hasattr(processor, "save_batch_results"):
                        processor.save_batch_results(
                            batch_idx, responses, {"batch_size": len(batch_items), "config": config}
                        )

                except Exception as e:
                    logger.error(f"Batch {batch_idx} processing failed: {e}")
                    responses = []
            except Exception as e:
                logger.error(f"Failed to create batch data for batch {batch_idx}: {e}")
                responses = []

        # Collect results
        if isinstance(responses, list):
            all_responses.extend(responses)
        elif responses is not None:
            all_responses.append(responses)

        if batch_data and batch_data.labels:
            all_ground_truth.extend(batch_data.labels)

    # Evaluate results
    if all_responses and all_ground_truth:
        try:
            logger.info("Evaluating results...")
            metrics = processor.evaluate_responses(all_responses, all_ground_truth)
            logger.info(f"Final metrics: {metrics}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            metrics = {}
    else:
        logger.warning("No valid responses or ground truth for evaluation")
        metrics = {}

    logger.success(f"Task {config.task} completed successfully!")
    return metrics


@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main(cfg: Config) -> None:
    """Main entry point."""
    try:
        asyncio.run(run_task(cfg))
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
