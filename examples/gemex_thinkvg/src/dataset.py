"""GEMeX-ThinkVG dataset loader.

Loads the GEMeX-ThinkVG dataset from HuggingFace and handles
MIMIC-CXR image path resolution.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from beartype import beartype
from datasets import load_dataset

from .schemas import parse_thinkvg_response


@beartype
class GEMeXDataset:
    """Dataset wrapper for GEMeX-ThinkVG from HuggingFace.

    Note: This dataset requires MIMIC-CXR-JPG images which must be
    downloaded separately from PhysioNet (requires credentialed access).

    Example:
        dataset = GEMeXDataset(
            mimic_cxr_root="/path/to/mimic-cxr-jpg",
            max_samples=1000,
        )
        for sample in dataset:
            image_path = sample["image_path"]
            question = sample["question"]
            answer = sample["answer"]  # Ground truth
            bbox = sample["bbox"]  # Ground truth bbox
    """

    QUESTION_TYPES = {
        "open_ended_questions": "open_ended",
        "closed_ended_questions": "closed_ended",
        "single_choice_questions": "single_choice",
        "multi_choice_questions": "multi_choice",
    }

    def __init__(
        self,
        mimic_cxr_root: Path | str | None = None,
        split: str = "train",
        max_samples: int | None = None,
        question_type: str | None = None,
    ) -> None:
        """Initialize GEMeX-ThinkVG dataset.

        Args:
            mimic_cxr_root: Root directory of MIMIC-CXR-JPG dataset.
                           If None, image_path will be relative paths.
            split: Dataset split (currently only "train" available)
            max_samples: Limit number of samples (None for all)
            question_type: Filter by question type:
                - "open_ended" / "open_ended_questions"
                - "closed_ended" / "closed_ended_questions"
                - "single_choice" / "single_choice_questions"
                - "multi_choice" / "multi_choice_questions"
                - None for all types

        Raises:
            ValueError: If question_type is invalid
        """
        self.mimic_cxr_root = Path(mimic_cxr_root) if mimic_cxr_root else None
        self.split = split

        # Load dataset
        self._dataset = load_dataset("BoKelvin/GEMeX-ThinkVG", split=split)

        # Filter by question type if specified
        if question_type:
            # Normalize question type
            normalized = self.QUESTION_TYPES.get(question_type, question_type)
            if normalized not in self.QUESTION_TYPES.values():
                raise ValueError(
                    f"Invalid question_type '{question_type}'. "
                    f"Must be one of: {list(self.QUESTION_TYPES.values())}"
                )
            # Filter dataset
            self._dataset = self._dataset.filter(
                lambda x: self.QUESTION_TYPES.get(x["question_type"], x["question_type"])
                == normalized
            )

        # Limit samples if specified
        if max_samples is not None:
            self._dataset = self._dataset.select(
                range(min(max_samples, len(self._dataset)))
            )

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
        return self._transform_sample(item)

    @beartype
    def _resolve_image_path(self, relative_path: str) -> Path | None:
        """Resolve MIMIC-CXR image path.

        Args:
            relative_path: Relative path from dataset (e.g., /p10/p10000032/...)

        Returns:
            Absolute path if mimic_cxr_root set, else None
        """
        if self.mimic_cxr_root is None:
            return None

        # Remove leading slash if present
        if relative_path.startswith("/"):
            relative_path = relative_path[1:]

        full_path = self.mimic_cxr_root / relative_path

        # Check if file exists
        if not full_path.exists():
            return None

        return full_path

    @beartype
    def _parse_ground_truth(self, response_text: str) -> dict[str, Any]:
        """Parse ground truth from response field.

        Args:
            response_text: Raw response text with XML structure

        Returns:
            Parsed ground truth with answer, location reference, and bbox
        """
        parsed = parse_thinkvg_response(response_text)

        if parsed is None:
            return {
                "answer": "",
                "location_ref": "",
                "bbox": [0, 0, 0, 0],
            }

        location = parsed.get("location", {})

        return {
            "answer": parsed.get("answer", ""),
            "location_reference": location.get("reference", ""),
            "bbox": location.get("bbox", [0, 0, 0, 0]),
        }

    @beartype
    def _extract_reasoning_bboxes(self, thinkvg_text: str) -> list[list[int]]:
        """Extract bounding boxes mentioned in ThinkVG reasoning.

        Args:
            thinkvg_text: ThinkVG reasoning text with embedded coordinates

        Returns:
            List of bounding boxes found in the reasoning
        """
        # Find all bbox patterns like [x1, y1, x2, y2]
        bbox_pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        matches = re.findall(bbox_pattern, thinkvg_text)

        return [[int(x) for x in match] for match in matches]

    @beartype
    def _transform_sample(self, item: dict[str, Any]) -> dict[str, Any]:
        """Transform raw HuggingFace sample to our format.

        Args:
            item: Raw dataset item

        Returns:
            Transformed sample with standardized keys
        """
        # Resolve image path
        image_path_str = item.get("image_path", "")
        image_path = self._resolve_image_path(image_path_str)

        # Parse ground truth from response
        response_text = item.get("response", "")
        ground_truth = self._parse_ground_truth(response_text)

        # Extract reasoning bboxes
        thinkvg_text = item.get("thinkVG", "")
        reasoning_bboxes = self._extract_reasoning_bboxes(thinkvg_text)

        # Normalize question type
        raw_q_type = item.get("question_type", "open_ended_questions")
        question_type = self.QUESTION_TYPES.get(raw_q_type, "open_ended")

        return {
            "image_path": image_path,
            "image_path_relative": image_path_str,
            "question": item.get("question", ""),
            "question_type": question_type,
            # Ground truth for verification
            "answer": ground_truth["answer"],
            "location_reference": ground_truth["location_reference"],
            "bbox": ground_truth["bbox"],
            # Full reasoning chain (for SFT or reference)
            "thinkvg_reasoning": thinkvg_text,
            "reasoning_bboxes": reasoning_bboxes,
            # Raw response for parsing validation
            "raw_response": response_text,
            "metadata": {
                "split": item.get("split", self.split),
                "question_type_raw": raw_q_type,
            },
        }

    @beartype
    def get_by_question_type(self, q_type: str) -> list[dict[str, Any]]:
        """Get all samples of a specific question type.

        Args:
            q_type: Question type to filter

        Returns:
            List of matching samples
        """
        normalized = self.QUESTION_TYPES.get(q_type, q_type)
        return [s for s in self if s["question_type"] == normalized]

    @beartype
    def get_samples_with_images(self) -> list[dict[str, Any]]:
        """Get only samples where image files exist.

        Returns:
            List of samples with valid image paths
        """
        return [s for s in self if s["image_path"] is not None]
