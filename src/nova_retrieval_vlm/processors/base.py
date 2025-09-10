"""Base processor interface for task-specific processing."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from beartype import beartype
from loguru import logger
from pydantic import BaseModel

from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import MetadataDict
from nova_retrieval_vlm.types import ModelResponse


class ProcessorConfig(BaseModel):
    """Configuration for processors."""

    task_name: str
    model_name: str
    batch_size: int = 8
    use_retrieval: bool = False
    retrieval_type: str = "bm25"
    output_dir: Path = Path("./runs")
    skip_existing: bool = False


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
