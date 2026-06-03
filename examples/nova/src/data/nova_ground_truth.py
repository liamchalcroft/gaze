"""NOVA ground truth loader for proper evaluation.

This module provides access to NOVA dataset ground truth annotations
from the CSV files in the NOVA repository. Supports both local CSV
directories and automatic download from HuggingFace.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype
from loguru import logger

HF_REPO_ID = "c-i-ber/Nova"


@dataclass
class GroundTruthLocalization:
    """Ground truth localization data."""

    bbox: tuple[
        float, float, float, float
    ]  # (x1, y1, x2, y2) - converted from dataset's (x, y, width, height)


@dataclass
class GroundTruth:
    """Complete ground truth data for a NOVA sample."""

    filename: str
    case_id: str
    scan_id: str
    caption: str
    clinical_history: str
    final_diagnosis: str
    localizations: list[GroundTruthLocalization]


class NovaGroundTruth:
    """Load and access NOVA ground truth data from CSV files."""

    @beartype
    def __init__(self, nova_dir: str = "~/Nova") -> None:
        """Initialize ground truth loader from a local directory.

        Args:
            nova_dir: Path to NOVA dataset directory containing CSV files
        """
        self.nova_dir = Path(nova_dir).expanduser()
        self._ground_truth: dict[str, GroundTruth] = {}
        self._load_data()

    @classmethod
    def from_huggingface(cls, repo_id: str = HF_REPO_ID) -> NovaGroundTruth:
        """Download ground truth CSVs from HuggingFace and load them.

        Args:
            repo_id: HuggingFace dataset repository ID

        Returns:
            Populated NovaGroundTruth instance
        """
        from huggingface_hub import hf_hub_download

        logger.info(f"Downloading ground truth CSVs from {repo_id}")
        instance = object.__new__(cls)
        instance._ground_truth = {}

        # Download CSVs to HF cache (cached after first download)
        captions_path = hf_hub_download(repo_id, "captions.csv", repo_type="dataset")
        metadata_path = hf_hub_download(repo_id, "case_metadata.csv", repo_type="dataset")
        bboxes_path = hf_hub_download(repo_id, "bboxes_gold.csv", repo_type="dataset")

        instance.nova_dir = Path(captions_path).parent
        instance._load_captions_from(Path(captions_path))
        instance._load_case_metadata_from(Path(metadata_path))
        instance._load_bboxes_from(Path(bboxes_path))

        logger.info(f"Loaded {len(instance._ground_truth)} ground truth samples from HuggingFace")
        return instance

    def _load_data(self) -> None:
        """Load all ground truth data from CSV files in nova_dir."""
        if not self.nova_dir.exists():
            raise FileNotFoundError(f"NOVA directory not found: {self.nova_dir}")

        self._load_captions_from(self.nova_dir / "captions.csv")
        self._load_case_metadata_from(self.nova_dir / "case_metadata.csv")
        self._load_bboxes_from(self.nova_dir / "bboxes_gold.csv")

        logger.info(f"Loaded {len(self._ground_truth)} ground truth samples")

    def _load_captions_from(self, captions_file: Path) -> None:
        """Load caption data from a captions CSV file."""
        if not captions_file.exists():
            raise FileNotFoundError(f"Captions file not found: {captions_file}")

        with open(captions_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                if filename not in self._ground_truth:
                    self._ground_truth[filename] = GroundTruth(
                        filename=filename,
                        case_id=row["case_id"],
                        scan_id=row["scan_id"],
                        caption=row["caption"],
                        clinical_history="",
                        final_diagnosis="",
                        localizations=[],
                    )
                else:
                    self._ground_truth[filename].caption = row["caption"]

    def _load_case_metadata_from(self, metadata_file: Path) -> None:
        """Load case metadata including diagnosis from a CSV file."""
        if not metadata_file.exists():
            raise FileNotFoundError(f"Case metadata file not found: {metadata_file}")

        with open(metadata_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            case_to_metadata = {}
            for row in reader:
                case_id = row["case_id"]
                case_to_metadata[case_id] = {
                    "clinical_history": row["clinical_history"],
                    "final_diagnosis": row["final_diagnosis"],
                }

        # Update ground truth with case metadata
        for gt in self._ground_truth.values():
            if gt.case_id in case_to_metadata:
                metadata = case_to_metadata[gt.case_id]
                gt.clinical_history = metadata["clinical_history"]
                gt.final_diagnosis = metadata["final_diagnosis"]

    def _load_bboxes_from(self, bboxes_file: Path) -> None:
        """Load bounding box data from a CSV file."""
        if not bboxes_file.exists():
            raise FileNotFoundError(f"Bounding boxes file not found: {bboxes_file}")

        with open(bboxes_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                if filename not in self._ground_truth:
                    continue  # Skip if no corresponding caption/case

                # Convert from (x, y, width, height) to (x1, y1, x2, y2) format
                x = float(row["x"])
                y = float(row["y"])
                width = float(row["width"])
                height = float(row["height"])

                bbox = GroundTruthLocalization(
                    bbox=(
                        x,  # x1
                        y,  # y1
                        x + width,  # x2
                        y + height,  # y2
                    )
                )
                self._ground_truth[filename].localizations.append(bbox)

    @beartype
    def get_ground_truth(self, filename: str) -> GroundTruth | None:
        """Get ground truth for a specific filename.

        Args:
            filename: Image filename (e.g., "case0183_001.png")

        Returns:
            GroundTruth data or None if not found
        """
        return self._ground_truth.get(filename)

    @beartype
    def get_ground_truth_by_subject_id(self, subject_id: int) -> GroundTruth:
        """Get ground truth by subject ID (matches prediction subject IDs).

        Args:
            subject_id: Numeric subject ID (0-indexed)

        Returns:
            GroundTruth data for the subject

        Raises:
            IndexError: If subject_id is out of range
        """
        filenames = list(self._ground_truth.keys())

        if not (0 <= subject_id < len(filenames)):
            raise IndexError(
                f"Subject ID {subject_id} out of range. "
                f"Valid range: 0-{len(filenames) - 1} ({len(filenames)} samples)"
            )

        filename = filenames[subject_id]
        return self._ground_truth[filename]

    @beartype
    def list_all_filenames(self) -> list[str]:
        """Get list of all available filenames."""
        return list(self._ground_truth.keys())

    @beartype
    def __len__(self) -> int:
        """Number of ground truth samples."""
        return len(self._ground_truth)
