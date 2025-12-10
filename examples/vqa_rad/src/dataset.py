"""VQA-RAD dataset loader using HuggingFace datasets.

Loads the VQA-RAD visual question answering dataset.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from beartype import beartype
from datasets import load_dataset
from PIL import Image


@beartype
class VQARadDataset:
    """Dataset wrapper for VQA-RAD from HuggingFace.

    Example:
        dataset = VQARadDataset(split="train")
        for sample in dataset:
            image_path = sample["image_path"]  # Temp file path
            question = sample["question"]
            answer = sample["answer"]  # Ground truth
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: int | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize VQA-RAD dataset.

        Args:
            split: Dataset split ("train" or "test")
            max_samples: Limit number of samples (None for all)
            cache_dir: Directory to cache image files (uses temp dir if None)

        Raises:
            ValueError: If split is invalid
        """
        if split not in ("train", "test"):
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'")

        self.split = split
        self._dataset = load_dataset("flaviagiammarino/vqa-rad", split=split)

        if max_samples is not None:
            self._dataset = self._dataset.select(range(min(max_samples, len(self._dataset))))

        # Set up image cache directory
        if cache_dir is not None:
            self._cache_dir = cache_dir
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = None
        else:
            self._temp_dir = tempfile.TemporaryDirectory()
            self._cache_dir = Path(self._temp_dir.name)

        # Cache for image paths
        self._image_cache: dict[int, Path] = {}

    def __del__(self) -> None:
        """Cleanup temporary directory if used."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self._dataset)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over dataset samples."""
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample by index."""
        item = self._dataset[idx]
        return self._transform_sample(idx, item)

    @beartype
    def _get_image_path(self, idx: int, image: Image.Image) -> Path:
        """Get or create cached image path.

        Args:
            idx: Sample index
            image: PIL Image object

        Returns:
            Path to saved image file
        """
        if idx in self._image_cache:
            return self._image_cache[idx]

        # Save image to cache directory
        image_path = self._cache_dir / f"vqa_rad_{idx}.png"
        image.save(image_path, format="PNG")
        self._image_cache[idx] = image_path

        return image_path

    @beartype
    def _transform_sample(self, idx: int, item: dict[str, Any]) -> dict[str, Any]:
        """Transform raw HuggingFace sample to our format.

        Args:
            idx: Sample index
            item: Raw dataset item

        Returns:
            Transformed sample with standardized keys
        """
        # Get image and save to file
        image: Image.Image = item["image"]
        image_path = self._get_image_path(idx, image)

        question = item.get("question", "")
        answer = item.get("answer", "")

        # Determine answer type (closed vs open)
        answer_lower = answer.lower().strip()
        is_closed = answer_lower in ("yes", "no")

        return {
            "image_path": image_path,
            "image_size": image.size,  # (width, height)
            "question": question,
            "answer": answer,  # Ground truth
            "answer_type": "closed" if is_closed else "open",
            "metadata": {
                "split": self.split,
                "index": idx,
            },
        }

    @beartype
    def get_closed_subset(self) -> list[dict[str, Any]]:
        """Get only closed (yes/no) questions.

        Returns:
            List of samples with closed questions
        """
        return [s for s in self if s["answer_type"] == "closed"]

    @beartype
    def get_open_subset(self) -> list[dict[str, Any]]:
        """Get only open-ended questions.

        Returns:
            List of samples with open questions
        """
        return [s for s in self if s["answer_type"] == "open"]
