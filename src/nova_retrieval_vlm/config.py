from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class RetrievalType(str, Enum):
    """Supported retrieval types with validation."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class RetrievalConfig(BaseModel):
    """Retrieval configuration with field validation."""

    type: RetrievalType = Field(
        default=RetrievalType.HYBRID,
        description="Retrieval method. Hybrid with RRF typically performs best",
    )
    top_k: int = Field(default=5, ge=1, le=100, description="Number of documents to retrieve")
    hybrid_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Balance between BM25 and dense retrieval (0=BM25 only, 1=dense only)",
    )


class ModelConfig(BaseModel):
    """Model configuration with parameter validation."""

    name: str = Field(
        default="opengvlab/internvl3-14b:free",
        min_length=1,
        description="Model identifier (OpenRouter format: provider/model:tier)",
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts for API calls"
    )
    timeout: int = Field(default=60, ge=1, le=300, description="Request timeout in seconds")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature for generation"
    )
    max_tokens: int = Field(default=1024, ge=1, le=8192, description="Maximum tokens to generate")

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name format."""
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        # Could add more specific validation for OpenRouter format
        return v.strip()


class PathsConfig(BaseModel):
    """Path configuration with validation and auto-creation."""

    data_dir: str = Field(default="./data/nova", description="Base directory for dataset")
    index_dir: str = Field(default="indexes", description="Directory for search indexes")
    output_dir: str = Field(default="./runs", description="Output directory for results")

    @field_validator("data_dir", "index_dir", "output_dir")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        """Validate and normalize paths."""
        if not v.strip():
            raise ValueError("Path cannot be empty")

        # Normalize path
        normalized = str(Path(v).resolve())
        return normalized

    def create_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        for path_str in [self.data_dir, self.index_dir, self.output_dir]:
            Path(path_str).mkdir(parents=True, exist_ok=True)


class TaskType(str, Enum):
    """Supported task types."""

    LOCALIZATION = "localization"
    CAPTION = "caption"
    DIAGNOSIS = "diagnosis"
    VISUALIZE = "visualize"


class VisualizationConfig(BaseModel):
    """Visualization configuration with validation."""

    num_samples: int = Field(default=5, ge=1, le=100, description="Number of samples to visualize")
    out_dir: str | None = Field(
        default=None, description="Output directory for visualizations (None for auto-generate)"
    )
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code execution"
    )
    overlay: bool = Field(default=False, description="Whether to overlay predictions on images")


class Config(BaseModel):
    """Main configuration with field validation."""

    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    retrieval: RetrievalConfig = Field(
        default_factory=RetrievalConfig, description="Retrieval configuration"
    )
    paths: PathsConfig = Field(default_factory=PathsConfig, description="Path configuration")
    task: TaskType = Field(default=TaskType.LOCALIZATION, description="Task to perform")
    batch_size: int = Field(
        default=4, ge=1, le=64, description="Number of samples per batch for inference"
    )
    use_retrieval: bool = Field(
        default=False, description="Whether to augment prompts with retrieved guideline passages"
    )
    prompt_text: str = Field(
        default="", description="Free-form text prompt for testing without image"
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig, description="Visualization configuration"
    )
    max_iterations: int = Field(
        default=5, ge=-1, description="Maximum iterations (≤0 for processing entire dataset)"
    )
    request_delay: float = Field(
        default=3.0,
        ge=0.0,
        le=60.0,
        description="Delay in seconds between API requests to avoid rate limiting",
    )

    @field_validator("prompt_text")
    @classmethod
    def validate_prompt_text(cls, v: str) -> str:
        """Validate and clean prompt text."""
        return v.strip()

    def model_post_init(self, __context) -> None:
        """Post-initialization setup."""
        # Auto-create directories
        self.paths.create_directories()

        # Validate task-specific configuration
        if self.task == TaskType.VISUALIZE and self.visualization.num_samples <= 0:
            raise ValueError("Visualization requires num_samples > 0")


