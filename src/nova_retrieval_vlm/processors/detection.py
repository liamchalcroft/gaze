"""Detection task processor (similar to localization but for general detection)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype import beartype
from PIL import Image

from nova_retrieval_vlm.evaluation.detection import evaluate_detection
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import ModelResponse
from nova_retrieval_vlm.types import parse_json_response

from .base import BaseProcessor


class DetectionProcessor(BaseProcessor):
    """Processor for detection tasks."""

    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process detection batch with real functionality."""
        responses = []

        # Initialize model adapter
        model_adapter = OpenAIAdapter(
            model_name=self.config.model_name, max_tokens=512, temperature=0.0
        )

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=False)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Load image
            Image.open(image_path)

            # Create detection prompt
            prompt = self._create_detection_prompt(image_path, metadata)

            # Get model response using existing OpenAI adapter interface
            response_text, generation_log = await model_adapter.generate(
                image_path=Path(image_path), passages=[], system_prompt=prompt
            )

            model_result = {
                "text": response_text,
                "confidence": 0.8,  # Default confidence
                "log": generation_log,
            }

            # Parse JSON response
            raw_text = model_result.get("text", "")

            # Parse structured response
            parsed_response = parse_json_response(raw_text)

            # Extract detections
            detections = parsed_response.get("detections", [])
            confidence = model_result.get("confidence", 0.5)

            # Create structured response
            response = ModelResponse(
                text=json.dumps({"detections": detections}),
                confidence=confidence,
                reasoning=parsed_response.get("reasoning", ""),
                metadata={
                    "image_path": str(image_path),
                    "num_detections": len(detections),
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                },
            )
            responses.append(response)

        return responses

    @beartype
    def evaluate_responses(
        self, responses: list[ModelResponse], ground_truth: list[str]
    ) -> EvaluationMetrics:
        """Evaluate detection responses."""
        # Extract predicted detections from responses
        predicted_detections = []
        for response in responses:
            parsed = json.loads(response.text)
            predicted_detections.append(parsed.get("detections", []))

        # Parse ground truth detections
        ground_truth_detections = []
        for gt in ground_truth:
            if isinstance(gt, str):
                gt_parsed = json.loads(gt)
                ground_truth_detections.append(gt_parsed.get("detections", []))
            else:
                ground_truth_detections.append(gt)

        # Use existing evaluation function
        results = evaluate_detection(predicted_detections, ground_truth_detections)

        return EvaluationMetrics(
            accuracy=results.get("mAP", 0.0),
            precision=results.get("precision"),
            recall=results.get("recall"),
            f1_score=results.get("f1_score"),
            auc_roc=results.get("auc"),
        )

    def _create_detection_prompt(self, image_path: Path, metadata: dict[str, Any]) -> str:
        """Create detection prompt for the medical image."""
        target_classes = metadata.get("target_classes", ["abnormality", "lesion", "tumor"])
        modality = metadata.get("modality", "medical image")

        prompt = f"""Detect and classify objects in this {modality}.

Image: {image_path.name}
Target classes: {", ".join(target_classes)}

Provide your response as JSON with the following format:
{{
    "detections": [
        {{
            "bbox": [x1, y1, x2, y2],  // Bounding box coordinates
            "class": "class_name",      // Object class
            "confidence": 0.95          // Detection confidence
        }},
        ...
    ],
    "reasoning": "Explanation of detections"
}}

Detect all instances of the target classes. Use [x1, y1, x2, y2] format for bounding boxes."""

        return prompt
