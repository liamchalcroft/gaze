"""Diagnosis task processor."""

from __future__ import annotations

from pathlib import Path

from beartype import beartype

from nova_retrieval_vlm.evaluation.diagnosis import evaluate_diagnosis_nova_official
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import DiagnosisMetrics
from nova_retrieval_vlm.types import ModelResponse

from .base import BaseProcessor


class DiagnosisProcessor(BaseProcessor):
    """Processor for diagnosis tasks."""

    # Only require diagnosis field for this processor
    REQUIRED_FIELDS: list[str] = ["diagnosis"]

    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process diagnosis batch."""
        responses = []

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=True)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Use common workflow helper (handles dimensions, prompt, adapter, JSON parsing)
            response_json = await self._get_parsed_model_response(
                image_path=Path(image_path),
                metadata=metadata,
                temperature=0.0,
            )

            # Extract and validate diagnosis data
            response = self._extract_diagnosis_response(
                response_json, image_path, batch_idx, i, metadata
            )
            responses.append(response)

        return responses

    @beartype
    def _extract_diagnosis_response(
        self,
        response_json: dict,
        image_path: str | Path,
        batch_idx: int,
        sample_idx: int,
        metadata: dict,
    ) -> ModelResponse:
        """Extract diagnosis-specific fields from parsed JSON response.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if "diagnosis" not in response_json:
            raise ValueError("Missing required 'diagnosis' key in response")

        diagnosis_data = response_json["diagnosis"]
        if not isinstance(diagnosis_data, dict):
            raise ValueError(f"Expected 'diagnosis' dict, got {type(diagnosis_data)}")
        if "primary_diagnosis" not in diagnosis_data:
            raise ValueError("Missing 'primary_diagnosis' in diagnosis response")
        if "confidence" not in diagnosis_data:
            raise ValueError("Missing 'confidence' in diagnosis response")

        diagnosis_text = diagnosis_data["primary_diagnosis"].strip()
        diagnosis = self._extract_primary_diagnosis(diagnosis_text)

        return ModelResponse(
            text=diagnosis,
            confidence=diagnosis_data["confidence"],
            reasoning=diagnosis_text,
            metadata={
                "image_path": str(image_path),
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
                "modality": metadata.get("modality", "unknown"),
                "full_response": diagnosis_text,
            },
        )

    @beartype
    def evaluate_responses(
        self, responses: list[ModelResponse], ground_truth: list[str]
    ) -> DiagnosisMetrics:
        """Evaluate diagnosis responses using NOVA protocol."""
        predictions = [r.text for r in responses]

        # Use NOVA official evaluation with LLM semantic matching
        results = evaluate_diagnosis_nova_official(predictions, ground_truth)

        return DiagnosisMetrics(
            top1=results["top1"],
            top5=results["top5"],
            coverage=results["coverage"],
            entropy=results["entropy"],
        )

    @beartype
    def _extract_primary_diagnosis(self, diagnosis_text: str) -> str:
        """Extract the primary diagnosis from the full response.

        The diagnosis text is expected to be the primary_diagnosis field from
        the structured JSON response, which should already be a clean diagnosis.

        Args:
            diagnosis_text: The primary diagnosis text from structured response

        Returns:
            The diagnosis string (cleaned)

        Raises:
            ValueError: If diagnosis text is empty
        """
        cleaned = diagnosis_text.strip()
        if not cleaned:
            raise ValueError("Empty diagnosis text provided")

        # The structured response already gives us the primary diagnosis,
        # just return it cleaned. If it contains a prefix, strip it.
        prefixes = ("primary diagnosis:", "diagnosis:", "most likely:")
        lower = cleaned.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        if not cleaned:
            raise ValueError("Diagnosis text was only a prefix with no content")

        return cleaned
