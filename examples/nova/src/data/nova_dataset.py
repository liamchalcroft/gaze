"""NOVA dataset loader for brain MRI analysis.

Loads the complete NOVA dataset from HuggingFace, using the parquet file
for metadata/ground truth and downloading images from the repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from beartype import beartype
from huggingface_hub import snapshot_download
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor

HF_REPO_ID = "c-i-ber/Nova"


class NovaDataset:
    """Complete NOVA dataset with images, metadata, and ground truth.

    Downloads the entire c-i-ber/Nova repository from HuggingFace on first use
    (cached for subsequent runs). The parquet file provides all metadata,
    ground truth, and image paths.
    """

    @beartype
    def __init__(
        self,
        data_dir: str | None = None,
        ground_truth_dir: str | None = None,
        transform: Compose | None = None,
    ) -> None:
        """Initialize complete NOVA dataset.

        Args:
            data_dir: Optional local directory override. If None, downloads
                      from HuggingFace (recommended).
            ground_truth_dir: Unused, kept for backward compatibility.
            transform: Optional torchvision transforms to apply to images.
        """
        self.transform = transform

        if data_dir and Path(data_dir).expanduser().exists():
            self._repo_dir = Path(data_dir).expanduser()
            logger.info(f"Using local NOVA data from {self._repo_dir}")
        else:
            logger.info(f"Downloading NOVA dataset from {HF_REPO_ID}...")
            self._repo_dir = Path(
                snapshot_download(HF_REPO_ID, repo_type="dataset")
            )
            logger.info(f"NOVA dataset cached at {self._repo_dir}")

        # Load the parquet which contains all metadata and ground truth
        parquet_path = self._repo_dir / "data" / "nova-v1.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Parquet file not found at {parquet_path}. "
                "Ensure the NOVA dataset was downloaded correctly."
            )
        self._df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(self._df)} samples from NOVA dataset")

    @beartype
    def __len__(self) -> int:
        return len(self._df)

    @beartype
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get complete sample with image, metadata, and ground truth."""
        if idx < 0 or idx >= len(self._df):
            raise IndexError(f"Index {idx} out of range [0, {len(self._df)})")

        row = self._df.iloc[idx]

        # Load image from repo
        image_path = self._repo_dir / row["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        filename = row["filename"]
        case_id = row["case_id"]
        scan_id = row["scan_id"]
        caption = row["caption_text"]

        # Extract metadata from the 'meta' dict
        meta = row["meta"] if isinstance(row["meta"], dict) else {}
        clinical_history = meta.get("clinical_history", "")
        final_diagnosis = meta.get("final_diagnosis", "")

        # Extract gold-standard bounding boxes
        bboxes_raw = row["bboxes"] if row["bboxes"] is not None else []
        localizations = []
        for bbox in bboxes_raw:
            if not isinstance(bbox, dict):
                continue
            if bbox.get("source") != "gold":
                continue
            x = float(bbox["x"])
            y = float(bbox["y"])
            w = float(bbox["width"])
            h = float(bbox["height"])
            # Convert (x, y, width, height) to (x1, y1, x2, y2)
            localizations.append({"bbox": (x, y, x + w, y + h)})

        return {
            "image": image,
            "metadata": {
                "image_id": idx,
                "filename": filename,
                "case_id": case_id,
                "scan_id": scan_id,
                "clinical_history": clinical_history,
                "final_diagnosis": final_diagnosis,
                "caption": caption,
                "localizations": localizations,
            },
            "ground_truth": {
                "filename": filename,
                "case_id": case_id,
                "scan_id": scan_id,
                "caption": caption,
                "clinical_history": clinical_history,
                "final_diagnosis": final_diagnosis,
                "localizations": localizations,
            },
            "hf_index": idx,
            "has_ground_truth": True,
        }


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
