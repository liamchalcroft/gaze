from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalConfig:
    type: str = "bm25"  # Options: bm25, dense, hybrid
    top_k: int = 5
    hybrid_ratio: float = 0.5


@dataclass
class ModelConfig:
    name: str = "opengvlab/internvl3-14b:free"  # Default OpenRouter model
    max_retries: int = 3
    timeout: int = 60
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class PathsConfig:
    data_dir: str = "./data/nova"  # Base directory for dataset
    index_dir: str = "indexes"
    output_dir: str = "./runs"  # Default output directory


@dataclass
class VisualizationConfig:
    num_samples: int = 5
    out_dir: Optional[str] = None
    trust_remote_code: bool = False
    overlay: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    task: str = "localization"  # Options: localization, caption, diagnosis, visualize
    batch_size: int = 4  # Number of samples per batch for inference
    use_retrieval: bool = False  # Whether to augment prompts with retrieved guideline passages
    prompt_text: str = ""  # Free-form text prompt for testing without image
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    max_iterations: int = 5  # Set to ≤0 for processing entire dataset
    request_delay: float = 3.0  # Delay in seconds between API requests to avoid rate limiting
    strict_mode: bool = True  # Whether to fail on non-critical errors
    approach: str = "baseline"  # Options: baseline, multiturn, visual_multiturn
    visual_rounds: int = 2  # Number of visual adjustment loops
