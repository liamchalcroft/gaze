"""NOVA benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path

from beartype import beartype


class TaskType(Enum):
    """Available task types for NOVA evaluation."""

    LOCALIZATION = "localization"
    CAPTION = "caption"
    DIAGNOSIS = "diagnosis"
    ALL = "all"  # Run all three tasks


@beartype
@dataclass
class NOVAConfig:
    """Configuration for NOVA benchmark evaluation.

    This is the main configuration for running NOVA benchmark evaluations.
    It configures both the radiant_harness and NOVA-specific settings.

    All parameters must be properly typed - no string coercion is performed.
    Use Path objects and TaskType enum directly.
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
        """Validate configuration and create output directory."""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate ranges
        if not 1 <= self.max_turns <= 20:
            raise ValueError("max_turns must be between 1 and 20")
        if not 1 <= self.batch_size <= 64:
            raise ValueError("batch_size must be between 1 and 64")


@dataclass(frozen=True)
class ConfidenceConfig:
    """Configuration for confidence score calculations."""

    base: float = 0.5
    comprehensive_bonus: float = 0.1
    per_evidence: float = 0.02
    per_differential: float = 0.02
    per_localization: float = 0.02
    per_tool_turn: float = 0.05
    max_bonus: float = 0.1
