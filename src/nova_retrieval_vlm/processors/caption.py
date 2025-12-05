"""Caption task processor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype

from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import CaptionMetrics
from nova_retrieval_vlm.types import ModelResponse

from .base import BaseProcessor


class CaptionProcessor(BaseProcessor):
    """Processor for caption tasks."""

    # Only require caption field for this processor
    REQUIRED_FIELDS: list[str] = ["caption"]

    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process caption batch."""
        responses = []

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=True)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Use common workflow helper (handles dimensions, prompt, adapter, JSON parsing)
            response_json = await self._get_parsed_model_response(
                image_path=Path(image_path),
                metadata=metadata,
                temperature=0.1,
            )

            # Extract and validate caption data
            response = self._extract_caption_response(
                response_json, image_path, batch_idx, i, metadata
            )
            responses.append(response)

        return responses

    @beartype
    def _extract_caption_response(
        self,
        response_json: dict[str, Any],
        image_path: str | Path,
        batch_idx: int,
        sample_idx: int,
        metadata: dict[str, Any],
    ) -> ModelResponse:
        """Extract caption-specific fields from parsed JSON response.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if "caption" not in response_json:
            raise ValueError("Missing required 'caption' key in response")

        caption_data = response_json["caption"]
        if not isinstance(caption_data, dict):
            raise ValueError(f"Expected 'caption' dict, got {type(caption_data)}")
        if "description" not in caption_data:
            raise ValueError("Missing 'description' in caption response")
        if "confidence" not in caption_data:
            raise ValueError("Missing 'confidence' in caption response")

        return ModelResponse(
            text=caption_data["description"].strip(),
            confidence=caption_data["confidence"],
            reasoning=f"Generated caption for {Path(image_path).name}",
            metadata={
                "image_path": str(image_path),
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
                "modality": metadata.get("modality", "unknown"),
            },
        )

    @beartype
    def evaluate_responses(
        self, responses: list[ModelResponse], ground_truth: list[str]
    ) -> CaptionMetrics:
        """Evaluate caption responses using NOVA caption metrics."""
        predictions = [r.text for r in responses]
        references = ground_truth

        # Lazy import to avoid torch import issues at module load
        from nova_retrieval_vlm.evaluation.caption import evaluate_caption

        results = evaluate_caption(predictions, references)

        return CaptionMetrics(
            bleu=results["bleu"],
            bert_f1=results["bert_f1"],
            meteor=results["meteor"],
            modality_f1=results["modality_f1"],
            clinical_f1=results["clinical_f1"],
            binary_accuracy=results["binary_accuracy"],
            binary_f1=results["binary_f1"],
            radgraph_f1=results["radgraph_f1"],
        )
