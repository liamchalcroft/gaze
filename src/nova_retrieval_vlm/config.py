from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration with parameter validation."""

    name: str = "x-ai/grok-4.1-fast:free"
    max_retries: int = 3
    timeout: int = 60
    temperature: float = 0.7
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Model name cannot be empty")
        if not (0 <= self.max_retries <= 10):
            raise ValueError("max_retries must be between 0 and 10")
        if not (1 <= self.timeout <= 300):
            raise ValueError("timeout must be between 1 and 300")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not (1 <= self.max_tokens <= 8192):
            raise ValueError("max_tokens must be between 1 and 8192")

        self.name = self.name.strip()


@dataclass
class PathsConfig:
    """Path configuration with validation and auto-creation."""

    data_dir: str = "./data/nova"
    output_dir: str = "./runs"

    def __post_init__(self) -> None:
        """Validate and normalize paths after initialization."""
        if not self.data_dir or not self.data_dir.strip():
            raise ValueError("data_dir cannot be empty")
        if not self.output_dir or not self.output_dir.strip():
            raise ValueError("output_dir cannot be empty")

        # Normalize paths
        self.data_dir = str(Path(self.data_dir).resolve())
        self.output_dir = str(Path(self.output_dir).resolve())

    def create_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        for path_str in [self.data_dir, self.output_dir]:
            Path(path_str).mkdir(parents=True, exist_ok=True)


class TaskType(str, Enum):
    """Supported task types."""

    LOCALIZATION = "localization"
    CAPTION = "caption"
    DIAGNOSIS = "diagnosis"
    VISUALIZE = "visualize"


@dataclass
class AgenticConfig:
    """Agentic processing configuration with enhanced research capabilities."""

    enabled: bool = False
    use_tools: bool = True
    max_turns: int = 10
    confidence_threshold: float = 0.7
    reasoning_enabled: bool = False
    enable_research_metrics: bool = True
    enabled_tools: list[str] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)
    single_shot: bool = False
    use_retrieval: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not (1 <= self.max_turns <= 20):
            raise ValueError("max_turns must be between 1 and 20")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")


@dataclass
class VisualizationConfig:
    """Visualization configuration with validation."""

    num_samples: int = 5
    out_dir: str | None = None
    trust_remote_code: bool = False
    overlay: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not (1 <= self.num_samples <= 100):
            raise ValueError("num_samples must be between 1 and 100")


@dataclass
class Config:
    """Main configuration with field validation."""

    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    task: TaskType = TaskType.LOCALIZATION
    batch_size: int = 4
    prompt_text: str = ""
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    max_iterations: int = 5
    request_delay: float = 3.0
    skip_existing: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate and clean prompt text
        self.prompt_text = self.prompt_text.strip()

        # Validate ranges
        if not (1 <= self.batch_size <= 64):
            raise ValueError("batch_size must be between 1 and 64")
        if self.request_delay < 0 or self.request_delay > 60.0:
            raise ValueError("request_delay must be between 0.0 and 60.0")

        # Auto-create directories
        self.paths.create_directories()

        # Validate task-specific configuration
        if self.task == TaskType.VISUALIZE and self.visualization.num_samples <= 0:
            raise ValueError("Visualization requires num_samples > 0")

    @property
    def output_dir(self) -> str:
        """Convenience property for output directory."""
        return self.paths.output_dir
