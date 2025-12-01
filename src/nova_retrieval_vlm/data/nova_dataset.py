"""NOVA dataset loader for brain MRI analysis.

This provides the complete NOVA dataset combining HuggingFace images with CSV ground truth and metadata.
"""

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

from nova_retrieval_vlm.data.nova_ground_truth import NovaGroundTruth

logger = logging.getLogger(__name__)


class NovaDataset:
    """Complete NOVA dataset with images, metadata, and ground truth."""

    def __init__(self, data_dir: str = "./data/nova", transform: Compose | None = None):
        """Initialize complete NOVA dataset.

        Args:
            data_dir: Path to cache or download dataset
            transform: torchvision transforms to apply to images
        """
        self.data_dir = data_dir
        self.transform = transform

        # Load HF dataset (images only)
        self.hf_dataset = load_dataset("c-i-ber/Nova", split="train")

        # Load CSV ground truth and metadata
        self.ground_truth = NovaGroundTruth("~/Nova")

        # Create HF index to ground truth mapping
        self._create_hf_to_gt_mapping()

        logger.info(f"Loaded {len(self.hf_dataset)} samples from NOVA complete dataset")

    def _create_hf_to_gt_mapping(self):
        """Create mapping from HF dataset indices to ground truth samples."""
        # Use the checksums from HF dataset to map to ground truth
        checksums = self.hf_dataset.info.download_checksums

        self.hf_to_gt_index = {}
        gt_filenames = list(self.ground_truth._ground_truth.keys())

        for hf_index, (filepath, info) in enumerate(checksums.items()):
            filename = filepath.split("/")[-1]
            if filename in gt_filenames:
                gt_index = gt_filenames.index(filename)
                self.hf_to_gt_index[hf_index] = gt_index
            else:
                # Fallback - this should not happen if datasets match
                self.hf_to_gt_index[hf_index] = None

        logger.info(
            f"Created mapping for {len([x for x in self.hf_to_gt_index.values() if x is not None])} samples"
        )

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get complete sample with image, metadata, and ground truth."""
        if idx < 0 or idx >= len(self.hf_dataset):
            raise IndexError(f"Index {idx} out of range [0, {len(self.hf_dataset)})")

        # Get HF sample (image only)
        hf_item = self.hf_dataset[idx]

        if "image" in hf_item and hf_item["image"] is not None:
            img = hf_item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            image = img.convert("RGB")
        else:
            raise ValueError(f"No image found for index {idx}")

        if self.transform:
            image = self.transform(image)

        # Get ground truth data using mapping
        gt_index = self.hf_to_gt_index.get(idx)

        if gt_index is not None:
            gt_sample = self.ground_truth.get_ground_truth_by_subject_id(gt_index)
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
                    "gt_index": gt_index,
                }
            else:
                # Fallback for missing ground truth
                return {
                    "image": image,
                    "metadata": {
                        "image_id": idx,
                        "filename": f"case_{idx:04d}.png",
                        "case_id": f"case_{idx:04d}",
                        "scan_id": "001",
                        "clinical_history": "",
                        "final_diagnosis": "",
                        "caption": "",
                        "localizations": [],
                    },
                    "ground_truth": {
                        "filename": f"case_{idx:04d}.png",
                        "case_id": f"case_{idx:04d}",
                        "scan_id": "001",
                        "caption": "",
                        "clinical_history": "",
                        "final_diagnosis": "",
                        "localizations": [],
                    },
                    "hf_index": idx,
                    "gt_index": None,
                }
        else:
            # No ground truth mapping available
            return {
                "image": image,
                "metadata": {
                    "image_id": idx,
                    "filename": f"case_{idx:04d}.png",
                    "case_id": f"case_{idx:04d}",
                    "scan_id": "001",
                    "clinical_history": "",
                    "final_diagnosis": "",
                    "caption": "",
                    "localizations": [],
                },
                "ground_truth": {
                    "filename": f"case_{idx:04d}.png",
                    "case_id": f"case_{idx:04d}",
                    "scan_id": "001",
                    "caption": "",
                    "clinical_history": "",
                    "final_diagnosis": "",
                    "localizations": [],
                },
                "hf_index": idx,
                "gt_index": None,
            }

    def get_dataloader(self, batch_size: int = 8, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader for the complete dataset.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the dataset

        Returns:
            DataLoader yielding batches of complete NOVA samples
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    @property
    def hf_dataset_ref(self) -> HFDataset:
        """Access the underlying HuggingFace dataset."""
        return self.hf_dataset

    def debug_sample(self, idx: int = 0) -> None:
        """Debug a sample to show complete structure."""
        sample = self[idx]

        print(f"DEBUG: Sample {idx}")
        print("=" * 50)
        print(f"HF Index: {sample['hf_index']}")
        print(f"GT Index: {sample['gt_index']}")
        print(f"Filename: {sample['metadata']['filename']}")
        print(f"Case ID: {sample['metadata']['case_id']}")
        print(f"Scan ID: {sample['metadata']['scan_id']}")
        print(f"Image shape: {sample['image'].size}")
        print(f"Clinical History: {sample['metadata']['clinical_history'][:200]}...")
        print(f"Final Diagnosis: {sample['metadata']['final_diagnosis']}")
        print(f"Caption: {sample['metadata']['caption'][:200]}...")
        print(f"Localizations: {len(sample['metadata']['localizations'])}")
        if sample["metadata"]["localizations"]:
            for i, loc in enumerate(sample["metadata"]["localizations"][:2]):
                print(f"  {i + 1}: {loc}")


def get_dataloader(batch_size: int, data_dir: str, use_transforms: bool = False) -> DataLoader:
    """Create a DataLoader for the NOVA dataset.

    Args:
        batch_size: Number of samples per batch
        data_dir: Path to cache or download dataset
        use_transforms: Whether to apply tensor transforms (for model inference)

    Returns:
        DataLoader yielding batches of complete NOVA samples
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
