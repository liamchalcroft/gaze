"""NOVA ground truth loader for proper evaluation.

This module provides access to NOVA dataset ground truth annotations
from the CSV files in the NOVA repository.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype
from loguru import logger


@dataclass
class GroundTruthLocalization:
    """Ground truth localization data."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) - converted from dataset's (x, y, width, height)


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
        """Initialize ground truth loader.

        Args:
            nova_dir: Path to NOVA dataset directory containing CSV files
        """
        self.nova_dir = Path(nova_dir).expanduser()
        self._ground_truth: dict[str, GroundTruth] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load all ground truth data from CSV files."""
        if not self.nova_dir.exists():
            raise FileNotFoundError(f"NOVA directory not found: {self.nova_dir}")

        # Load captions and case metadata
        self._load_captions()
        self._load_case_metadata()
        self._load_bboxes()

        logger.info(f"Loaded {len(self._ground_truth)} ground truth samples")

    def _load_captions(self) -> None:
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

    def _load_case_metadata(self) -> None:
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

    def _load_bboxes(self) -> None:
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

                # Convert from (x, y, width, height) to (x1, y1, x2, y2) format
                x = float(row["x"])
                y = float(row["y"])
                width = float(row["width"])
                height = float(row["height"])

                bbox = GroundTruthLocalization(
                    bbox=(
                        x,           # x1
                        y,           # y1
                        x + width,   # x2
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
