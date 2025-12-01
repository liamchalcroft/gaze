"""Caption task processor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype
from PIL import Image

# Lazy import to avoid torch import issues during collection
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.prompts.prompt_loader import create_enhanced_prompt
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import ModelResponse

from .base import BaseProcessor


class CaptionProcessor(BaseProcessor):
    """Processor for caption tasks."""

    @beartype
    async def _parse_json_with_retry(
        self, raw_text: str, image_path: Path, system_prompt: str, max_retries: int = 3
    ) -> dict[str, Any] | None:
        """Parse JSON response with simple retry logic.

        Same implementation as LocalizationProcessor for consistency.
        """
        import json
        import re

        from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
        from nova_retrieval_vlm.schemas import NOVA_UNIFIED_SCHEMA

        for attempt in range(max_retries + 1):
            try:
                # Attempt to extract JSON from text (more reliable than direct parse)
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if json_match:
                    response_json = json.loads(json_match.group())
                    if "caption" in response_json and response_json["caption"]:
                        if attempt > 0:
                            self.logger.info(f"JSON parsing succeeded on attempt {attempt + 1}")
                        return response_json
                    else:
                        if attempt == max_retries:
                            self.logger.error(
                                "JSON parsed but missing caption field after all attempts"
                            )
                            return None
                        self.logger.warning("JSON parsed but missing caption field, will retry")
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
                    f"Retrying JSON generation for caption (attempt {attempt + 2}/{max_retries + 1})"
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
                    system_prompt=f'{system_prompt}\n\nCRITICAL: Your entire response MUST be valid JSON with the exact structure: {{"caption": {{"description": "...", "confidence": 0.8}}}}',
                    max_tokens=4096,
                    temperature=0.0,
                    response_format=NOVA_UNIFIED_SCHEMA,
                )
                raw_text = retry_response

        # All attempts failed
        return None

    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process caption batch with real functionality."""
        responses = []

        # Initialize model adapter
        model_adapter = OpenAIAdapter(model_name=self.config.model_name)

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=False)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Load image
            Image.open(image_path)

            # Create unified prompt for all tasks
            prompt = create_enhanced_prompt(
                template_name="all_tasks.jinja",
                image_path=image_path,
                passages=[],
                metadata={
                    **metadata,
                    "width": Image.open(image_path).width,
                    "height": Image.open(image_path).height,
                    "image_id": image_path.name,
                    "enable_visual_tools": False,
                    "enable_web_search": False,
                },
                mode="single_turn",
            )

            # Get model response using existing OpenAI adapter interface
            response_text, generation_log = await model_adapter.generate(
                image_path=Path(image_path),
                passages=[],
                system_prompt=prompt,
                max_tokens=4096,  # Increased for comprehensive response
                temperature=0.1,
            )

            model_result = {
                "text": response_text,
                "confidence": 0.8,  # Default confidence
                "log": generation_log,
            }

            # Extract caption from JSON response with retry logic
            response_text = model_result.get("text", "").strip()

            # Parse JSON response from unified prompt with retry
            response_json = await self._parse_json_with_retry(
                response_text, Path(image_path), prompt
            )

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

            caption = response_json.get("caption", {}).get("description", "").strip()
            confidence = response_json.get("caption", {}).get("confidence", 0.5)

            # Create structured response
            response = ModelResponse(
                text=caption,
                confidence=confidence,
                reasoning=f"Generated caption for {image_path.name}",
                metadata={
                    "image_path": str(image_path),
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "modality": metadata.get("modality", "unknown"),
                },
            )
            responses.append(response)

        return responses

    @beartype
    def _create_caption_prompt(self, image_path: Path, metadata: dict[str, Any]) -> str:
        """Create caption prompt for testing purposes."""
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
        """Evaluate caption responses."""
        predictions = [r.text for r in responses]
        references = ground_truth

        # Use existing evaluation function
        # Lazy import to avoid torch import issues
        from nova_retrieval_vlm.evaluation.caption import evaluate_caption

        results = evaluate_caption(predictions, references)

        return EvaluationMetrics(
            accuracy=results.get("bleu", 0.0),  # Use BLEU as primary accuracy metric
            precision=results.get("bert_precision"),
            recall=results.get("bert_recall"),
            f1_score=results.get("bert_f1"),
            auc_roc=results.get("meteor"),
        )
