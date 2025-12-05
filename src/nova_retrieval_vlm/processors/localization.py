"""Localization task processor."""

from __future__ import annotations

import json
from pathlib import Path

from beartype import beartype
from PIL import Image

from nova_retrieval_vlm.evaluation.detection import evaluate_detection
from nova_retrieval_vlm.schemas import NOVA_UNIFIED_SCHEMA
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import DetectionMetrics
from nova_retrieval_vlm.types import ModelResponse
from nova_retrieval_vlm.visual_reasoning.image_ops import adjust_contrast

from .base import BaseProcessor


class LocalizationProcessor(BaseProcessor):
    """Processor for localization tasks."""

    # Require localization field for this processor
    REQUIRED_FIELDS: list[str] = ["localization"]

    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process localization batch with real functionality."""
        responses = []

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=True)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Load and preprocess image with proper resource management
            with Image.open(image_path) as image:
                # Apply image processing based on metadata hints
                if metadata.get("low_contrast", False):
                    processed_image = adjust_contrast(image, 1.5)
                    try:
                        width, height = processed_image.width, processed_image.height
                    finally:
                        # Close derived image to prevent memory leak
                        processed_image.close()
                else:
                    width, height = image.width, image.height

            # Create unified prompt using shared method
            prompt = self._create_unified_prompt(
                image_path=Path(image_path),
                metadata=metadata,
                width=width,
                height=height,
            )

            # Get model response using shared adapter with structured outputs
            response_text, _ = await self.adapter.generate(
                image_path=Path(image_path),
                passages=[],  # No passages for basic localization
                system_prompt=prompt,
                max_tokens=4096,
                temperature=0.0,
                response_format=NOVA_UNIFIED_SCHEMA,
            )

            # Parse JSON response with retry logic for structured outputs
            try:
                response_json = await self._parse_json_with_retry(
                    response_text, Path(image_path), prompt
                )
            except JSONParseError as e:
                self.logger.error(f"JSON parsing failed: {e}")
                raise

            # Extract ALL three tasks from unified response - fail fast on missing required fields
            if "localization" not in response_json:
                raise ValueError("Missing required 'localization' key in response")

            caption_data = response_json.get("caption", {})
            diagnosis_data = response_json.get("diagnosis", {})
            localization_data = response_json["localization"]

            # Extract localization data - validate structure
            if "localizations" not in localization_data:
                raise ValueError("Missing required 'localizations' key in localization response")
            localizations = localization_data["localizations"]

            # Validate each localization entry has required fields
            boxes = []
            labels = []
            for loc_idx, loc in enumerate(localizations):
                if "bounding_box" not in loc:
                    raise ValueError(f"Missing 'bounding_box' in localization entry {loc_idx}")
                if "finding" not in loc:
                    raise ValueError(f"Missing 'finding' in localization entry {loc_idx}")
                boxes.append(loc["bounding_box"])
                labels.append(loc["finding"])

            # Extract confidence from localization data (average of all localizations)
            loc_confidences = [loc.get("confidence", 0.0) for loc in localizations]
            confidence = sum(loc_confidences) / len(loc_confidences) if loc_confidences else 0.0

            # Create structured response
            # Save the COMPLETE unified response with all three tasks
            unified_response = {
                "caption": caption_data,
                "diagnosis": diagnosis_data,
                "localization": {"boxes": boxes, "labels": labels, "localizations": localizations},
            }

            response = ModelResponse(
                text=json.dumps(unified_response),
                confidence=confidence,
                reasoning=f"Processed unified response with {len(localizations)} localizations",
                metadata={
                    "image_path": str(image_path),
                    "num_boxes": len(boxes),
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "has_caption": bool(caption_data),
                    "has_diagnosis": bool(diagnosis_data),
                },
            )
            responses.append(response)

        return responses

    @beartype
    def evaluate_responses(
        self, responses: list[ModelResponse], ground_truth: list[str]
    ) -> DetectionMetrics:
        """Evaluate localization responses using NOVA detection metrics.

        Raises:
            ValueError: If JSON parsing fails or required fields are missing.
        """
        # Extract predicted boxes from responses - format for evaluator
        predicted_dicts = []
        for response in responses:
            try:
                parsed = json.loads(response.text)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in response during evaluation: {response.text[:100]}"
                ) from e

            # Response must have localization key from process_batch
            if "localization" not in parsed:
                raise ValueError(f"Missing 'localization' key in response: {parsed}")

            loc_data = parsed["localization"]
            if "localizations" not in loc_data:
                raise ValueError(f"Missing 'localizations' in localization data: {loc_data}")

            localizations = loc_data["localizations"]
            boxes = []
            for loc in localizations:
                if "bounding_box" not in loc:
                    raise ValueError(f"Missing 'bounding_box' in localization: {loc}")
                boxes.append(loc["bounding_box"])

            # Create dict format expected by evaluator
            predicted_dicts.append(
                {
                    "boxes": boxes,
                    "scores": [1.0] * len(boxes),  # Default confidence scores
                    "labels": [0] * len(boxes),  # Default label 0 for all detections
                }
            )

        # Parse ground truth boxes - format for evaluator
        ground_truth_dicts = []
        for gt in ground_truth:
            gt_parsed = json.loads(gt)
            if "boxes" not in gt_parsed:
                raise ValueError(f"Ground truth missing 'boxes' key: {gt[:100]}")
            boxes = gt_parsed["boxes"]

            # Create dict format expected by evaluator
            ground_truth_dicts.append(
                {
                    "boxes": boxes,
                    "scores": [1.0] * len(boxes),  # Ground truth has confidence 1.0
                    "labels": [0] * len(boxes),  # Default label 0 for all ground truth
                }
            )

        # Use existing evaluation function
        results = evaluate_detection(predicted_dicts, ground_truth_dicts)

        return DetectionMetrics(
            map30=results["map30"],
            map50=results["map50"],
            map50_95=results["map50_95"],
            acc50=results["acc50"],
            tp30=results["tp30"],
            fp30=results["fp30"],
        )
