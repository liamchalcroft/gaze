"""Caption task processor."""

from __future__ import annotations

from pathlib import Path

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
                max_tokens=1024,  # Increased for comprehensive response
                temperature=0.1,
            )

            model_result = {
                "text": response_text,
                "confidence": 0.8,  # Default confidence
                "log": generation_log,
            }

            # Extract caption from JSON response
            import json

            response_text = model_result.get("text", "").strip()

            # Parse JSON response from unified prompt
            response_json = json.loads(response_text)
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
