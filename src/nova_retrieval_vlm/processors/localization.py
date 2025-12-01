"""Localization task processor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype import beartype
from PIL import Image

from nova_retrieval_vlm.evaluation.detection import evaluate_detection
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.prompts.prompt_loader import create_enhanced_prompt
from nova_retrieval_vlm.schemas import NOVA_UNIFIED_SCHEMA
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import ModelResponse
from nova_retrieval_vlm.visual_reasoning.image_ops import adjust_contrast

from .base import BaseProcessor


class LocalizationProcessor(BaseProcessor):
    """Processor for localization tasks."""

    @beartype
    async def _parse_json_with_retry(
        self, raw_text: str, image_path: Path, system_prompt: str, max_retries: int = 3
    ) -> dict[str, Any] | None:
        """Parse JSON response with simple retry logic.

        Args:
            raw_text: Raw text response from model
            image_path: Path to the image (for retry)
            system_prompt: System prompt used (for retry)
            max_retries: Maximum number of retry attempts

        Returns:
            Parsed JSON response or None if all attempts fail
        """
        import json
        import re

        for attempt in range(max_retries + 1):
            try:
                # Attempt to extract JSON from text (more reliable than direct parse)
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if json_match:
                    response_json = json.loads(json_match.group())
                    if self._validate_json_response(response_json):
                        if attempt > 0:
                            self.logger.info(f"JSON parsing succeeded on attempt {attempt + 1}")
                        return response_json
                    else:
                        if attempt == max_retries:
                            self.logger.error(
                                "JSON parsed but missing required fields after all attempts"
                            )
                            return None
                        self.logger.warning("JSON parsed but missing required fields, will retry")
                else:
                    if attempt == max_retries:
                        self.logger.error("No JSON found in response after all attempts")
                        return None
                    self.logger.warning("No JSON found in response, will retry")
            except json.JSONDecodeError as e:
                if attempt == max_retries:
                    self.logger.error(f"JSON decode failed after all attempts: {e}")
                    return None
                self.logger.error(f"JSON decode failed (attempt {attempt + 1}): {e}, will retry")

            # Retry with new generation
            if attempt < max_retries:
                self.logger.info(
                    f"Retrying JSON generation (attempt {attempt + 2}/{max_retries + 1})"
                )

                # Re-initialize model adapter for retry
                model_adapter = OpenAIAdapter(
                    model_name=self.config.model_name,
                    reasoning_enabled=self.config.reasoning_enabled,
                    reasoning_effort=self.config.reasoning_effort,
                    enable_caching=self.config.enable_caching,
                )

                retry_response, retry_log = await model_adapter.generate(
                    image_path=image_path,
                    _passages=[],
                    system_prompt=f'{system_prompt}\n\nCRITICAL: Your entire response MUST be valid JSON with the exact structure: {{"caption": {{...}}, "diagnosis": {{...}}, "localization": {{...}}}}',
                    max_tokens=4096,  # Default for retry
                    temperature=0.0,  # Low temperature for structured output
                    response_format=NOVA_UNIFIED_SCHEMA,
                )
                raw_text = retry_response

        # All attempts failed
        return None

    def _validate_json_response(self, response_json: dict[str, Any]) -> bool:
        """Validate that JSON response has required NOVA structure."""
        required_top_level = ["caption", "diagnosis", "localization"]
        return all(key in response_json for key in required_top_level)

    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process localization batch with real functionality."""
        responses = []

        # Initialize model adapter with reasoning enabled
        self.logger.info(
            f"Initializing OpenAIAdapter with reasoning_enabled={self.config.reasoning_enabled}, effort={self.config.reasoning_effort}, caching={self.config.enable_caching}"
        )
        model_adapter = OpenAIAdapter(
            model_name=self.config.model_name,
            reasoning_enabled=self.config.reasoning_enabled,
            reasoning_effort=self.config.reasoning_effort,
            enable_caching=self.config.enable_caching,
        )

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
                image_path=Path(image_path),  # Convert string to Path
                passages=[],
                metadata={
                    **metadata,
                    "width": image.width,
                    "height": image.height,
                    "image_id": Path(image_path).name,  # Convert to Path first
                    "enable_visual_tools": False,
                    "enable_web_search": False,
                },
                mode="single_turn",
            )

            # Get model response using structured outputs for better reliability
            response_text, generation_log = await model_adapter.generate(
                image_path=Path(image_path),
                _passages=[],  # No passages for basic localization (note underscore)
                system_prompt=prompt,
                max_tokens=4096,  # Increased for comprehensive response
                temperature=0.0,
                response_format=NOVA_UNIFIED_SCHEMA,
            )

            model_result = {
                "text": response_text,
                "confidence": 0.8,  # Default confidence
                "log": generation_log,
            }

            # Parse JSON response with retry logic for structured outputs
            raw_text = model_result.get("text", "")
            response_json = await self._parse_json_with_retry(raw_text, Path(image_path), prompt)

            # Handle parsing failure
            if response_json is None:
                self.logger.error("Failed to parse JSON response after all retry attempts")
                # Return ModelResponse with failure indication
                failure_response = ModelResponse(
                    text="JSON parsing failed after multiple attempts",
                    confidence=0.0,
                    reasoning="Failed to parse JSON response after all retry attempts",
                    metadata={
                        "image_path": str(image_path),
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "error": "json_parse_failure",
                    },
                )
                responses.append(failure_response)
                continue

            # Extract ALL three tasks from unified response
            caption_data = response_json.get("caption", {})
            diagnosis_data = response_json.get("diagnosis", {})
            localization_data = response_json.get("localization", {})

            # Extract localization data
            localizations = localization_data.get("localizations", [])
            boxes = [loc.get("bounding_box", []) for loc in localizations]
            labels = [loc.get("finding", "") for loc in localizations]
            confidence = model_result.get("confidence", 0.5)

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
    def _create_localization_prompt(self, image_path: Path, metadata: dict[str, Any]) -> str:
        """Create localization prompt for testing purposes."""
        from nova_retrieval_vlm.prompts.prompt_loader import create_enhanced_prompt

        return create_enhanced_prompt(
            template_name="all_tasks.jinja",
            image_path=image_path,
            passages=[],
            metadata={
                **metadata,
                "width": 1024,  # Default for testing
                "height": 1024,
                "image_id": image_path.name,
                "enable_visual_tools": False,
                "enable_web_search": False,
            },
            mode="single_turn",
        )

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
                # Handle unified response format with nested localization
                if "localization" in parsed:
                    loc_data = parsed.get("localization", {})
                    localizations = loc_data.get("localizations", [])
                    boxes = [
                        loc.get("bounding_box", [])
                        for loc in localizations
                        if loc.get("bounding_box")
                    ]
                else:
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
                # Handle invalid JSON gracefully - treat as empty prediction
                self.logger.warning(
                    f"Invalid JSON in response, treating as empty: {response.text[:100]}"
                )
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
