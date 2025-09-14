"""Base processor interface for task-specific processing."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from beartype import beartype
from loguru import logger
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import MetadataDict
from nova_retrieval_vlm.types import ModelResponse


class ProcessorConfig(BaseModel):
    """Configuration for processors with field validation."""

    task_name: str = Field(..., min_length=1, description="Name of the processing task")
    model_name: str = Field(..., min_length=1, description="Model identifier for processing")
    batch_size: int = Field(
        default=8, ge=1, le=128, description="Number of samples to process in each batch"
    )
    use_retrieval: bool = Field(default=False, description="Whether to use retrieval augmentation")
    retrieval_type: str = Field(
        default="bm25", pattern="^(bm25|dense|hybrid)$", description="Type of retrieval to use"
    )
    output_dir: Path = Field(
        default=Path("./runs"), description="Directory to save processing results"
    )
    skip_existing: bool = Field(
        default=False, description="Whether to skip processing of existing results"
    )

    @field_validator("task_name", "model_name")
    @classmethod
    def validate_names(cls, v: str) -> str:
        """Validate and clean name fields."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Name cannot be empty")
        return cleaned

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Validate and create output directory."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def model_post_init(self, __context) -> None:
        """Post-initialization validation."""
        # Task-specific validation
        valid_tasks = {"localization", "caption", "diagnosis", "detection"}
        if self.task_name not in valid_tasks:
            logger.warning(f"Task '{self.task_name}' not in standard tasks: {valid_tasks}")

        # Retrieval validation
        if self.use_retrieval and self.retrieval_type not in ["bm25", "dense", "hybrid"]:
            raise ValueError(f"Invalid retrieval_type: {self.retrieval_type}")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True  # Allow Path objects


class BaseProcessor(ABC):
    """Base class for task-specific processors."""

    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config
        self.logger = logger.bind(task=config.task_name)

    @abstractmethod
    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process a batch of data and return model responses."""
        ...

    @abstractmethod
    @beartype
    def evaluate_responses(
        self, responses: list[ModelResponse], ground_truth: list[str]
    ) -> EvaluationMetrics:
        """Evaluate model responses against ground truth."""
        ...

    @beartype
    def should_skip_batch(self, batch_idx: int) -> bool:
        """Check if batch should be skipped (for resuming)."""
        if not self.config.skip_existing:
            return False

        output_file = self.config.output_dir / f"batch_{batch_idx}.json"
        return output_file.exists()

    @beartype
    def save_batch_results(
        self, batch_idx: int, responses: list[ModelResponse], metadata: MetadataDict
    ) -> None:
        """Save batch results to disk."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        output_file = self.config.output_dir / f"batch_{batch_idx}.json"

        results = {
            "batch_idx": batch_idx,
            "task": self.config.task_name,
            "model": self.config.model_name,
            "responses": [r.model_dump() for r in responses],
            "metadata": metadata,
        }

        with output_file.open("w") as f:
            import json

            json.dump(results, f, indent=2)

        self.logger.debug(f"Saved batch {batch_idx} to {output_file}")

    @beartype
    def load_batch_results(self, batch_idx: int) -> list[ModelResponse]:
        """Load saved batch results."""
        output_file = self.config.output_dir / f"batch_{batch_idx}.json"

        if not output_file.exists():
            raise FileNotFoundError(f"No saved results for batch {batch_idx}")

        with output_file.open("r") as f:
            import json

            data = json.load(f)

        return [ModelResponse(**r) for r in data["responses"]]
