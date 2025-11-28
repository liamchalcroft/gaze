"""NOVA dataset loader for brain MRI analysis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from datasets import Dataset as HFDataset
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor

logger = logging.getLogger(__name__)


class NovaDataset:
    """PyTorch Dataset for the NOVA brain-MRI HuggingFace dataset (test split only)."""

    def __init__(self, data_dir: str, transform: Compose | None = None):
        """Initialize dataset.

        Args:
            data_dir: Path to cache or download dataset.
            transform: torchvision transforms to apply to images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.dataset = load_dataset("Ano-2090/Nova", split="test", cache_dir=self.data_dir)
        logger.info("Loaded NOVA test split with %d samples.", len(self.dataset))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError(f"Index {idx} out of range [0, {len(self.dataset)})")

        item = self.dataset[idx]

        if "image" in item and item["image"] is not None:
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            image = img.convert("RGB")
        else:
            image_path = item.get("image_path")
            if not image_path:
                raise ValueError(f"No image or image_path for index {idx}")
            image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "metadata": item.get("metadata", {}),
            "image_id": item.get("image_id", idx),
        }

    @property
    def hf_dataset(self) -> HFDataset:
        """Access the underlying HuggingFace dataset for reference data."""
        return self.dataset


def get_dataloader(batch_size: int, data_dir: str, use_transforms: bool = False) -> DataLoader:
    """Create a DataLoader for the NOVA dataset.

    Args:
        batch_size: Number of samples per batch.
        data_dir: Path to cache or download dataset (test split only).
        use_transforms: Whether to apply tensor transforms (for local model inference).

    Returns:
        DataLoader yielding batches of dicts.
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")

    transforms = None
    if use_transforms:
        transforms = Compose(
            [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    dataset = NovaDataset(data_dir=data_dir, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
