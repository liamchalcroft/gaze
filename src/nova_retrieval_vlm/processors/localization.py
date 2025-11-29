"""Localization task processor."""

from __future__ import annotations

import json
from pathlib import Path

from beartype import beartype
from PIL import Image

from nova_retrieval_vlm.evaluation.detection import evaluate_detection
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.prompts.prompt_loader import create_enhanced_prompt
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import ModelResponse
from nova_retrieval_vlm.visual_reasoning.image_ops import adjust_contrast

from .base import BaseProcessor


class LocalizationProcessor(BaseProcessor):
    """Processor for localization tasks."""

    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process localization batch with real functionality."""
        responses = []

        # Initialize model adapter
        model_adapter = OpenAIAdapter(model_name=self.config.model_name)

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=False)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Load and preprocess image
            image = Image.open(image_path)

            # Apply image processing based on metadata hints
            if metadata.get("low_contrast", False):
                image = adjust_contrast(image, 1.5)

            # Create unified prompt for all tasks
            prompt = create_enhanced_prompt(
                template_name="all_tasks.jinja",
                image_path=image_path,
                passages=[],
                metadata={
                    **metadata,
                    "width": image.width,
                    "height": image.height,
                    "image_id": image_path.name,
                    "enable_visual_tools": False,
                    "enable_web_search": False,
                },
                mode="single_turn",
            )

            # Get model response using existing OpenAI adapter interface
            response_text, generation_log = await model_adapter.generate(
                image_path=Path(image_path),
                passages=[],  # No passages for basic localization
                system_prompt=prompt,
                max_tokens=1024,  # Increased for comprehensive response
                temperature=0.0,
            )

            model_result = {
                "text": response_text,
                "confidence": 0.8,  # Default confidence
                "log": generation_log,
            }

            # Parse JSON response
            raw_text = model_result.get("text", "")

            # Parse the unified JSON response
            import json

            response_json = json.loads(raw_text)
            # Extract localization data from unified prompt response
            localization_data = response_json.get("localization", {})
            parsed_response = {
                "boxes": [
                    loc.get("bounding_box", [])
                    for loc in localization_data.get("localizations", [])
                ],
                "labels": [
                    loc.get("finding", "") for loc in localization_data.get("localizations", [])
                ],
                "reasoning": f"Extracted {len(localization_data.get('localizations', []))} localizations",
            }

            # Extract bounding boxes
            boxes = parsed_response.get("boxes", [])
            labels = parsed_response.get("labels", [])
            confidence = model_result.get("confidence", 0.5)

            # Create structured response
            response = ModelResponse(
                text=json.dumps({"boxes": boxes, "labels": labels}),
                confidence=confidence,
                reasoning=parsed_response.get("reasoning", ""),
                metadata={
                    "image_path": str(image_path),
                    "num_boxes": len(boxes),
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
        """Evaluate localization responses."""
        # Extract predicted boxes from responses - format for evaluator
        predicted_dicts = []
        for response in responses:
            try:
                parsed = json.loads(response.text)
                boxes = parsed.get("boxes", [])
                # Create dict format expected by evaluator
                predicted_dicts.append(
                    {
                        "boxes": boxes,
                        "scores": [1.0] * len(boxes),  # Default confidence scores
                        "labels": [0] * len(boxes),  # Default label 0 for all detections
                    }
                )
            except json.JSONDecodeError:
                # Handle invalid JSON by using empty detection dict
                predicted_dicts.append({"boxes": [], "scores": [], "labels": []})

        # Parse ground truth boxes - format for evaluator
        ground_truth_dicts = []
        for gt in ground_truth:
            if isinstance(gt, str):
                gt_parsed = json.loads(gt)
                boxes = gt_parsed.get("boxes", [])
            else:
                boxes = gt if isinstance(gt, list) else []

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

        return EvaluationMetrics(
            accuracy=results.get("map50", 0.0),  # Use mAP@0.5 as primary accuracy
            precision=results.get("precision"),
            recall=results.get("recall"),
            f1_score=results.get("f1_score"),
            auc_roc=results.get("map30"),  # Use mAP@0.3 as secondary metric
        )
