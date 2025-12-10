"""PubmedQA dataset loader using HuggingFace datasets.

Loads the PubmedQA dataset and provides a simple interface for iteration.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from beartype import beartype
from datasets import load_dataset


@beartype
class PubmedQADataset:
    """Dataset wrapper for PubmedQA from HuggingFace.

    Example:
        dataset = PubmedQADataset(config="pqa_labeled", split="train")
        for sample in dataset:
            question = sample["question"]
            context = sample["context"]
            answer = sample["answer"]  # Ground truth
    """

    VALID_CONFIGS = {"pqa_labeled", "pqa_artificial", "pqa_unlabeled"}

    def __init__(
        self,
        config: str = "pqa_labeled",
        split: str = "train",
        max_samples: int | None = None,
    ) -> None:
        """Initialize PubmedQA dataset.

        Args:
            config: Dataset configuration:
                - "pqa_labeled": 1k expert-annotated samples (recommended)
                - "pqa_artificial": 211k machine-generated samples
                - "pqa_unlabeled": 61k unlabeled samples
            split: Dataset split (usually "train")
            max_samples: Limit number of samples (None for all)

        Raises:
            ValueError: If config is invalid
        """
        if config not in self.VALID_CONFIGS:
            raise ValueError(
                f"Invalid config '{config}'. Must be one of: {self.VALID_CONFIGS}"
            )

        self.config = config
        self.split = split
        self._dataset = load_dataset("qiaojin/PubMedQA", config, split=split)

        if max_samples is not None:
            self._dataset = self._dataset.select(range(min(max_samples, len(self._dataset))))

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self._dataset)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over dataset samples."""
        for item in self._dataset:
            yield self._transform_sample(item)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample by index."""
        return self._transform_sample(self._dataset[idx])

    @beartype
    def _transform_sample(self, item: dict[str, Any]) -> dict[str, Any]:
        """Transform raw HuggingFace sample to our format.

        Args:
            item: Raw dataset item

        Returns:
            Transformed sample with standardized keys
        """
        context_data = item.get("context", {})

        return {
            "pubid": item.get("pubid"),
            "question": item.get("question", ""),
            "context": context_data.get("contexts", []),
            "labels": context_data.get("labels", []),
            "meshes": context_data.get("meshes", []),
            "long_answer": item.get("long_answer", ""),
            "answer": item.get("final_decision", ""),  # Ground truth: yes/no/maybe
            "metadata": {
                "pubid": item.get("pubid"),
                "config": self.config,
            },
        }
