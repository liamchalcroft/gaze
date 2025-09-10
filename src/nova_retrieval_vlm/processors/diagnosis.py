"""Diagnosis task processor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype
from PIL import Image

from nova_retrieval_vlm.evaluation.diagnosis import evaluate_diagnosis_nova_official
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
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
        model_adapter = OpenAIAdapter(
            model_name=self.config.model_name,
            max_tokens=512,
            temperature=0.0,  # Use deterministic generation for diagnosis
        )

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=False)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Load image
            Image.open(image_path)

            # Create diagnosis prompt
            prompt = self._create_diagnosis_prompt(image_path, metadata)

            # Get model response using existing OpenAI adapter interface
            response_text, generation_log = await model_adapter.generate(
                image_path=Path(image_path), passages=[], system_prompt=prompt
            )

            model_result = {
                "text": response_text,
                "confidence": 0.8,  # Default confidence
                "log": generation_log,
            }

            # Extract diagnosis
            diagnosis_text = model_result.get("text", "").strip()
            confidence = model_result.get("confidence", 0.5)

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

    def _create_diagnosis_prompt(self, image_path: Path, metadata: dict[str, Any]) -> str:
        """Create diagnosis prompt for the medical image."""
        modality = metadata.get("modality", "medical image")
        patient_info = metadata.get("patient_info", "")
        clinical_history = metadata.get("clinical_history", "")

        prompt = f"""Analyze this {modality} and provide a medical diagnosis.

Image: {image_path.name}
{f"Patient information: {patient_info}" if patient_info else ""}
{f"Clinical history: {clinical_history}" if clinical_history else ""}

Based on your analysis, provide:
1. Primary diagnosis (most likely condition)
2. Differential diagnoses (alternative possibilities)
3. Confidence level and reasoning
4. Recommended follow-up or additional studies

Primary Diagnosis:"""

        return prompt

    def _extract_primary_diagnosis(self, diagnosis_text: str) -> str:
        """Extract the primary diagnosis from the full response."""
        lines = diagnosis_text.split("\n")

        # Look for lines starting with diagnosis indicators
        for line in lines:
            line = line.strip()
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
