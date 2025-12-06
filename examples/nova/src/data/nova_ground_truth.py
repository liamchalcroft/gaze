"""NOVA ground truth loader for proper evaluation.

This module provides access to NOVA dataset ground truth annotations
from the CSV files in the NOVA repository.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class GroundTruthLocalization:
    """Ground truth localization data."""

    bbox: tuple[float, float, float, float]  # (x, y, width, height)


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

    def __init__(self, nova_dir: str = "~/Nova"):
        """Initialize ground truth loader.

        Args:
            nova_dir: Path to NOVA dataset directory containing CSV files
        """
        self.nova_dir = Path(nova_dir).expanduser()
        self._ground_truth: dict[str, GroundTruth] = {}
        self._load_data()

    def _load_data(self):
        """Load all ground truth data from CSV files."""
        if not self.nova_dir.exists():
            raise FileNotFoundError(f"NOVA directory not found: {self.nova_dir}")

        # Load captions and case metadata
        self._load_captions()
        self._load_case_metadata()
        self._load_bboxes()

        logger.info(f"Loaded {len(self._ground_truth)} ground truth samples")

    def _load_captions(self):
        """Load caption data from captions.csv."""
        captions_file = self.nova_dir / "captions.csv"
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

    def _load_case_metadata(self):
        """Load case metadata including diagnosis from case_metadata.csv."""
        metadata_file = self.nova_dir / "case_metadata.csv"
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

    def _load_bboxes(self):
        """Load bounding box data from bboxes_gold.csv."""
        bboxes_file = self.nova_dir / "bboxes_gold.csv"
        if not bboxes_file.exists():
            raise FileNotFoundError(f"Bounding boxes file not found: {bboxes_file}")

        with open(bboxes_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                if filename not in self._ground_truth:
                    continue  # Skip if no corresponding caption/case

                bbox = GroundTruthLocalization(
                    bbox=(
                        float(row["x"]),
                        float(row["y"]),
                        float(row["width"]),
                        float(row["height"]),
                    )
                )
                self._ground_truth[filename].localizations.append(bbox)

    def get_ground_truth(self, filename: str) -> GroundTruth | None:
        """Get ground truth for a specific filename.

        Args:
            filename: Image filename (e.g., "case0183_001.png")

        Returns:
            GroundTruth data or None if not found
        """
        return self._ground_truth.get(filename)

    def get_ground_truth_by_subject_id(self, subject_id: int) -> GroundTruth | None:
        """Get ground truth by subject ID (matches prediction subject IDs).

        This attempts to map numeric subject IDs to filenames.
        """
        # Convert subject_id to expected filename format
        # Subject IDs in predictions might be 0-based indices
        filenames = list(self._ground_truth.keys())

        if 0 <= subject_id < len(filenames):
            filename = filenames[subject_id]
            gt = self._ground_truth[filename]
            return gt

        return None

    def list_all_filenames(self) -> list[str]:
        """Get list of all available filenames."""
        return list(self._ground_truth.keys())

    def __len__(self) -> int:
        """Number of ground truth samples."""
        return len(self._ground_truth)

    def __iter__(self):
        """Iterate over ground truth samples."""
        return iter(self._ground_truth.values())
