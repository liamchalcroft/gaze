"""Diagnosis task processor."""

from __future__ import annotations

from pathlib import Path

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
                max_tokens=1024,  # Increased for comprehensive response
                temperature=0.0,
            )

            model_result = {
                "text": response_text,
                "confidence": 0.8,  # Default confidence
                "log": generation_log,
            }

            # Extract diagnosis from JSON response
            import json

            response_text = model_result.get("text", "").strip()

            # Parse JSON response from unified prompt
            response_json = json.loads(response_text)
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
