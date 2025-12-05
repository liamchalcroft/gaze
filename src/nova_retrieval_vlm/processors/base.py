"""Base processor interface for task-specific processing."""

from __future__ import annotations

import json
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import JSONParseError
from nova_retrieval_vlm.types import MetadataDict
from nova_retrieval_vlm.types import ModelResponse
from nova_retrieval_vlm.types import parse_json_response


class ProcessorConfig(BaseModel):
    """Configuration for processors with field validation."""

    task_name: str = Field(..., min_length=1, description="Name of the processing task")
    model_name: str = Field(..., min_length=1, description="Model identifier for processing")
    batch_size: int = Field(
        default=8, ge=1, le=128, description="Number of samples to process in each batch"
    )
    output_dir: Path = Field(
        default=Path("./runs"), description="Directory to save processing results"
    )
    skip_existing: bool = Field(
        default=False, description="Whether to skip processing of existing results"
    )
    reasoning_enabled: bool = Field(
        default=False, description="Whether to enable model reasoning (e.g., Grok reasoning mode)"
    )
    reasoning_effort: str = Field(
        default="high",
        pattern="^(high|medium|low|minimal|none)$",
        description="Reasoning effort level for supported models",
    )
    enable_caching: bool = Field(
        default=True, description="Enable prompt caching for consistent system prompts"
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, _context: object) -> None:
        """Post-initialization validation."""
        # Task-specific validation
        # 'all_tasks' = unified task (captioning, diagnosis, localization)
        valid_tasks = {"localization", "caption", "diagnosis", "detection", "all_tasks"}
        if self.task_name not in valid_tasks:
            logger.warning(f"Task '{self.task_name}' not in standard tasks: {valid_tasks}")


class BaseProcessor(ABC):
    """Base class for task-specific processors."""

    # Required fields for JSON validation - subclasses MUST override
    REQUIRED_FIELDS: list[str] = []  # Abstract - subclasses define their requirements

    @beartype
    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config
        self.logger = logger.bind(task=config.task_name)
        # Import here to avoid circular dependency, type annotation for clarity
        from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
        self._adapter: OpenAIAdapter | None = None  # Lazy-initialized adapter

    @property
    def adapter(self) -> OpenAIAdapter:
        """Lazy-initialize and return the model adapter."""
        if self._adapter is None:
            # Already imported in __init__ for type annotation
            from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
            self._adapter = OpenAIAdapter(
                model_name=self.config.model_name,
                reasoning_enabled=self.config.reasoning_enabled,
                reasoning_effort=self.config.reasoning_effort,
                enable_caching=self.config.enable_caching,
            )
        return self._adapter

    @beartype
    def _create_unified_prompt(
        self,
        image_path: Path,
        metadata: dict[str, Any],
        width: int = 1024,
        height: int = 1024,
        mode: str = "single_turn",
    ) -> str:
        """Create unified prompt for all tasks.

        Args:
            image_path: Path to the image file
            metadata: Additional metadata dict
            width: Image width in pixels
            height: Image height in pixels
            mode: Prompt mode ('single_turn' or 'multi_turn')

        Returns:
            Formatted prompt string
        """
        from nova_retrieval_vlm.prompts.prompt_loader import create_enhanced_prompt

        return create_enhanced_prompt(
            template_name="all_tasks.jinja",
            image_path=image_path,
            passages=[],
            metadata={
                **metadata,
                "width": width,
                "height": height,
                "image_id": image_path.name,
                "enable_visual_tools": False,
                "enable_web_search": False,
            },
            mode=mode,
        )

    @beartype
    async def _parse_json_with_retry(
        self,
        raw_text: str,
        image_path: Path,
        system_prompt: str,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Parse JSON response with retry logic.

        Args:
            raw_text: Raw text response from model
            image_path: Path to the image (for retry)
            system_prompt: System prompt used (for retry)
            max_retries: Maximum number of retry attempts

        Returns:
            Parsed JSON response

        Raises:
            JSONParseError: If all parsing attempts fail
        """
        from nova_retrieval_vlm.schemas import NOVA_UNIFIED_SCHEMA

        for attempt in range(max_retries + 1):
            try:
                # Use the proper JSON parsing utility from types.py
                response_json = parse_json_response(raw_text)
                if self._validate_json_response(response_json):
                    if attempt > 0:
                        self.logger.info(f"JSON parsing succeeded on attempt {attempt + 1}")
                    return response_json

                if attempt == max_retries:
                    raise JSONParseError(
                        raw_text,
                        "JSON parsed but missing required fields",
                        attempt + 1,
                    )
                self.logger.warning("JSON parsed but missing required fields, retrying")

            except JSONParseError:
                if attempt == max_retries:
                    raise
                self.logger.warning(f"JSON parse failed (attempt {attempt + 1}), retrying")

            # Retry with new generation using shared adapter
            if attempt < max_retries:
                self.logger.info(
                    f"Retrying JSON generation (attempt {attempt + 2}/{max_retries + 1})"
                )

                required_structure = (
                    "{" + ", ".join(f'"{f}": {{...}}' for f in self.REQUIRED_FIELDS) + "}"
                )
                json_hint = f"CRITICAL: Response MUST be valid JSON: {required_structure}"
                retry_response, _ = await self.adapter.generate(
                    image_path=image_path,
                    passages=[],
                    system_prompt=f"{system_prompt}\n\n{json_hint}",
                    max_tokens=4096,
                    temperature=0.0,
                    response_format=NOVA_UNIFIED_SCHEMA,
                )
                raw_text = retry_response

        # This point is unreachable - loop always raises on final attempt
        # Added for type checker satisfaction
        raise AssertionError("Unreachable: loop should have raised JSONParseError")

    @beartype
    def _validate_json_response(self, response_json: dict[str, Any]) -> bool:
        """Validate that JSON response has required fields."""
        return all(key in response_json for key in self.REQUIRED_FIELDS)

    @staticmethod
    @beartype
    def _get_image_dimensions(image_path: str | Path) -> tuple[int, int]:
        """Load image and return (width, height) dimensions.

        Uses context manager for proper resource cleanup.
        """
        from PIL import Image

        with Image.open(image_path) as image:
            return image.width, image.height

    @beartype
    async def _get_parsed_model_response(
        self,
        image_path: Path,
        metadata: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Common workflow: create prompt, get model response, parse JSON.

        This helper handles the standard single-turn processing workflow:
        1. Get image dimensions
        2. Create unified prompt
        3. Call model adapter
        4. Parse and validate JSON response

        Args:
            image_path: Path to the image file
            metadata: Metadata for prompt creation
            temperature: Model temperature (default 0.0 for determinism)

        Returns:
            Parsed and validated JSON response dict

        Raises:
            JSONParseError: If JSON parsing fails after retries
        """
        # Get image dimensions
        width, height = self._get_image_dimensions(image_path)

        # Create unified prompt
        prompt = self._create_unified_prompt(
            image_path=image_path,
            metadata=metadata,
            width=width,
            height=height,
        )

        # Get model response
        response_text, _ = await self.adapter.generate(
            image_path=image_path,
            passages=[],
            system_prompt=prompt,
            max_tokens=4096,
            temperature=temperature,
        )

        # Parse JSON with retry
        return await self._parse_json_with_retry(
            response_text.strip(), image_path, prompt
        )

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
            json.dump(results, f, indent=2)

        self.logger.debug(f"Saved batch {batch_idx} to {output_file}")

    @beartype
    def load_batch_results(self, batch_idx: int) -> list[ModelResponse]:
        """Load saved batch results."""
        output_file = self.config.output_dir / f"batch_{batch_idx}.json"

        if not output_file.exists():
            raise FileNotFoundError(f"No saved results for batch {batch_idx}")

        with output_file.open("r") as f:
            data = json.load(f)

        return [ModelResponse(**r) for r in data["responses"]]
