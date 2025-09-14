"""Caption task processor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype
from PIL import Image

# Lazy import to avoid torch import issues during collection
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import ModelResponse

from .base import BaseProcessor


class CaptionProcessor(BaseProcessor):
    """Processor for caption tasks."""

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

            # Create caption prompt
            prompt = self._create_caption_prompt(image_path, metadata)

            # Get model response using existing OpenAI adapter interface
            response_text, generation_log = await model_adapter.generate(
                image_path=Path(image_path),
                passages=[],
                system_prompt=prompt,
                max_tokens=256,
                temperature=0.1,
            )

            model_result = {
                "text": response_text,
                "confidence": 0.8,  # Default confidence
                "log": generation_log,
            }

            # Extract caption text
            caption = model_result.get("text", "").strip()
            confidence = model_result.get("confidence", 0.5)

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

    def _create_caption_prompt(self, image_path: Path, metadata: dict[str, Any]) -> str:
        """Create caption prompt for the medical image."""
        modality = metadata.get("modality", "medical image")
        patient_info = metadata.get("patient_info", "")

        prompt = f"""Provide a detailed medical description of this {modality}.

Image: {image_path.name}
{f"Patient context: {patient_info}" if patient_info else ""}

Describe:
- Image modality and acquisition details
- Anatomical structures visible
- Any abnormalities, lesions, or pathological findings
- Image quality and technical parameters

Provide a comprehensive but concise medical caption."""

        return prompt
