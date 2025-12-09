"""NOVA dataset loader for brain MRI analysis.

Provides the complete NOVA dataset combining HuggingFace images with CSV ground truth and metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from datasets import Dataset as HFDataset
from datasets import load_dataset
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor

from .nova_ground_truth import NovaGroundTruth


class NovaDataset:
    """Complete NOVA dataset with images, metadata, and ground truth."""

    @beartype
    def __init__(
        self,
        data_dir: str = "./data/nova",
        ground_truth_dir: str | None = None,
        transform: Compose | None = None,
    ) -> None:
        """Initialize complete NOVA dataset."""
        self.data_dir = data_dir
        self.transform = transform

        hf_ds = load_dataset("c-i-ber/Nova", split="train")
        if not isinstance(hf_ds, HFDataset):
            raise TypeError(f"Expected Dataset, got {type(hf_ds).__name__}")
        self.hf_dataset: HFDataset = hf_ds

        gt_path = ground_truth_dir if ground_truth_dir else data_dir
        self.ground_truth = NovaGroundTruth(gt_path)
        logger.info(f"Loaded {len(self.hf_dataset)} samples from NOVA complete dataset")

    @beartype
    def __len__(self) -> int:
        return len(self.hf_dataset)

    @beartype
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get complete sample with image, metadata, and ground truth."""
        if idx < 0 or idx >= len(self.hf_dataset):
            raise IndexError(f"Index {idx} out of range [0, {len(self.hf_dataset)})")

        hf_item = self.hf_dataset[idx]
        if "image" not in hf_item or hf_item["image"] is None:
            raise ValueError(f"No image found for index {idx}")

        img = hf_item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))

        # Safe filename extraction - check for None explicitly
        filename_attr = getattr(img, "filename", None)
        if not filename_attr:
            raise ValueError(f"Image at index {idx} missing filename attribute")
        filename = Path(filename_attr).name
        image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        gt_sample = self.ground_truth.get_ground_truth(filename)
        if gt_sample:
            return {
                "image": image,
                "metadata": {
                    "image_id": idx,
                    "filename": gt_sample.filename,
                    "case_id": gt_sample.case_id,
                    "scan_id": gt_sample.scan_id,
                    "clinical_history": gt_sample.clinical_history,
                    "final_diagnosis": gt_sample.final_diagnosis,
                    "caption": gt_sample.caption,
                    "localizations": [{"bbox": loc.bbox} for loc in gt_sample.localizations],
                },
                "ground_truth": {
                    "filename": gt_sample.filename,
                    "case_id": gt_sample.case_id,
                    "scan_id": gt_sample.scan_id,
                    "caption": gt_sample.caption,
                    "clinical_history": gt_sample.clinical_history,
                    "final_diagnosis": gt_sample.final_diagnosis,
                    "localizations": [{"bbox": loc.bbox} for loc in gt_sample.localizations],
                },
                "hf_index": idx,
                "has_ground_truth": True,
            }

        raise KeyError(f"No ground truth found for filename {filename} (HF index {idx})")


@beartype
def default_transforms(mean: list[float] | None = None, std: list[float] | None = None) -> Compose:
    """Create default torchvision transforms for NOVA dataset images."""
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    return Compose([ToTensor(), Normalize(mean=mean, std=std)])


@beartype
def get_dataloader(
    data_dir: str,
    batch_size: int = 8,
    shuffle: bool = False,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create a DataLoader with default transforms for the NOVA dataset."""
    transforms = default_transforms(mean=mean, std=std)
    dataset = NovaDataset(data_dir=data_dir, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
