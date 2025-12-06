"""NOVA benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path


class TaskType(Enum):
    """Available task types for NOVA evaluation."""

    LOCALIZATION = "localization"
    CAPTION = "caption"
    DIAGNOSIS = "diagnosis"
    ALL = "all"  # Run all three tasks


@dataclass
class NOVAConfig:
    """Configuration for NOVA benchmark evaluation.

    This is the main configuration for running NOVA benchmark evaluations.
    It configures both the radiant_harness and NOVA-specific settings.
    """

    # Model settings (passed to radiant_harness)
    model_name: str = "openai/gpt-4o"
    max_turns: int = 10
    use_tools: bool = True
    use_web_search: bool = False
    reasoning_enabled: bool = False
    reasoning_effort: str = "high"

    # NOVA-specific settings
    task: TaskType = TaskType.ALL
    data_dir: Path = field(default_factory=lambda: Path("./data/nova"))
    output_dir: Path = field(default_factory=lambda: Path("./runs"))
    batch_size: int = 4
    skip_existing: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.task, str):
            self.task = TaskType(self.task)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate ranges
        if not 1 <= self.max_turns <= 20:
            raise ValueError("max_turns must be between 1 and 20")
        if not 1 <= self.batch_size <= 64:
            raise ValueError("batch_size must be between 1 and 64")
