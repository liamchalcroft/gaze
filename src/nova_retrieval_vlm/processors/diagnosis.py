"""Diagnosis task processor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype
from PIL import Image

from nova_retrieval_vlm.evaluation.diagnosis import evaluate_diagnosis_nova_official
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.prompts.prompt_loader import create_enhanced_prompt
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import ModelResponse

from .base import BaseProcessor


class DiagnosisProcessor(BaseProcessor):
    """Processor for diagnosis tasks."""

    @beartype
    async def _parse_json_with_retry(
        self, raw_text: str, image_path: Path, system_prompt: str, max_retries: int = 3
    ) -> dict[str, Any]:
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
                    if "diagnosis" in response_json and response_json["diagnosis"]:
                        if attempt > 0:
                            self.logger.info(f"JSON parsing succeeded on attempt {attempt + 1}")
                        return response_json
                    else:
                        if attempt == max_retries:
                            self.logger.error(
                                "JSON parsed but missing diagnosis field after all attempts"
                            )
                            return None
                        self.logger.warning("JSON parsed but missing diagnosis field, will retry")
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
                    f"Retrying JSON generation for diagnosis (attempt {attempt + 2}/{max_retries + 1})"
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
                    passages=[],
                    system_prompt=f'{system_prompt}\n\nCRITICAL: Your entire response MUST be valid JSON with the exact structure: {{"diagnosis": {{"primary_diagnosis": "...", "confidence": 0.8}}}}',
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    response_format=NOVA_UNIFIED_SCHEMA,
                )
                raw_text = retry_response

        # All attempts failed
        return None

    @beartype
    def _create_diagnosis_prompt(self, image_path: Path, metadata: dict[str, Any]) -> str:
        """Create diagnosis prompt for testing purposes."""
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
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process diagnosis batch with real functionality."""
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
                temperature=0.0,
            )

            model_result = {
                "text": response_text,
                "confidence": 0.8,  # Default confidence
                "log": generation_log,
            }

            # Extract diagnosis from JSON response

            response_text = model_result.get("text", "").strip()

            # Parse JSON response from unified prompt with retry
            response_json = await self._parse_json_with_retry(
                response_text, Path(image_path), prompt
            )

            # Handle parsing failure
            if response_json is None:
                self.logger.error("Failed to parse JSON response after all retry attempts")
                # Return minimal response with failure indication
                response = ModelResponse(
                    text="JSON parsing failed after multiple attempts",
                    confidence=0.0,  # Minimum confidence for failed parsing
                    reasoning="JSON parsing failed after multiple attempts",
                    metadata={
                        "image_path": str(image_path),
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "modality": metadata.get("modality", "unknown"),
                    },
                )
                responses.append(response)
                continue

            diagnosis_text = response_json.get("diagnosis", {}).get("primary_diagnosis", "").strip()
            confidence = response_json.get("diagnosis", {}).get("confidence", 0.5)

            # Parse structured diagnosis if provided
            diagnosis = self._extract_primary_diagnosis(diagnosis_text)

            # Create structured response
            response = ModelResponse(
                text=diagnosis,
                confidence=confidence,
                reasoning=diagnosis_text,  # Full reasoning in reasoning field
                metadata={
                    "image_path": str(image_path),
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "modality": metadata.get("modality", "unknown"),
                    "full_response": diagnosis_text,
                },
            )
            responses.append(response)

        return responses

    @beartype
    def evaluate_responses(
        self, responses: list[ModelResponse], ground_truth: list[str]
    ) -> EvaluationMetrics:
        """Evaluate diagnosis responses."""
        predictions = [r.text for r in responses]

        # Use existing evaluation function
        results = evaluate_diagnosis_nova_official(predictions, ground_truth)

        return EvaluationMetrics(
            accuracy=results.get("accuracy", 0.0),
            precision=results.get("precision"),
            recall=results.get("recall"),
            f1_score=results.get("f1_score"),
            auc_roc=results.get("auc"),  # If available from evaluation
        )

    def _extract_primary_diagnosis(self, diagnosis_text: str) -> str:
        """Extract the primary diagnosis from the full response."""
        lines = diagnosis_text.split("\n")

        # Look for lines starting with diagnosis indicators
        for raw_line in lines:
            line = raw_line.strip()
            if line.lower().startswith(("primary diagnosis:", "diagnosis:", "most likely:")):
                # Extract the diagnosis after the colon
                parts = line.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
            elif line and not line.startswith(("1.", "2.", "3.", "4.", "-", "•")):
                # If no explicit marker, take the first substantial line
                if len(line) > 10:  # Avoid short fragments
                    return line

        # Fallback: return first line or original text
        return lines[0].strip() if lines else diagnosis_text
