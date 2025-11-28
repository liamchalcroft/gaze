"""Agentic localization processor.

Enhanced localization with visual reasoning integration,
tool calling, and multi-turn refinement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger

from nova_retrieval_vlm.agentic.processor import AgenticProcessor
from nova_retrieval_vlm.agentic.processor import AgenticResult
from nova_retrieval_vlm.agentic.retrieval_manager import RetrievalManager
from nova_retrieval_vlm.evaluation.detection import evaluate_detection
from nova_retrieval_vlm.processors.base import BaseProcessor
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import ModelResponse


class AgenticLocalizationProcessor(BaseProcessor):
    """Agentic processor for localization tasks.

    Extends base localization with:
    - Pre-analysis visual reasoning (structure detection, symmetry)
    - Tool calling for interactive analysis (zoom, crop, contrast)
    - Multi-turn refinement for uncertain cases
    - Retrieval augmentation support
    """

    def __init__(
        self,
        config: Any,
        use_visual_reasoning: bool = True,
        use_tools: bool = True,
        max_turns: int = 3,
        index_dir: Path | str | None = None,
    ):
        """Initialize agentic localization processor.

        Args:
            config: ProcessorConfig with model_name, output_dir, etc.
            use_visual_reasoning: Enable visual pre-analysis
            use_tools: Enable tool calling (zoom, crop, etc.)
            max_turns: Max turns for multi-turn refinement
            index_dir: Directory containing retrieval indexes
        """
        super().__init__(config)
        self.use_visual_reasoning = use_visual_reasoning
        self.use_tools = use_tools
        self.max_turns = max_turns

        self._agentic_processor = AgenticProcessor(
            model_name=config.model_name,
            use_visual_reasoning=use_visual_reasoning,
            use_tools=use_tools,
            max_turns=max_turns,
        )

        # Initialize retrieval manager if enabled
        self._retrieval_manager = None
        if config.use_retrieval and index_dir:
            self._retrieval_manager = RetrievalManager(
                index_dir=index_dir,
                retrieval_type=config.retrieval_type,
                top_k=5,
            )

    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process localization batch with agentic analysis."""
        responses = []

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=False)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Get retrieval passages if configured
            retrieval_passages = self._get_retrieval_passages(metadata)

            # Run agentic analysis
            result = await self._agentic_processor.analyze(
                image_path=Path(image_path),
                task="localization",
                metadata=metadata,
                retrieval_passages=retrieval_passages,
            )

            # Convert to ModelResponse
            response = self._convert_result(result, image_path, batch_idx, i)
            responses.append(response)

            # Log analysis details
            self._log_analysis(result, i)

        return responses

    def _get_retrieval_passages(
        self,
        metadata: dict[str, Any],
        visual_analysis: Any = None,
    ) -> list[str]:
        """Get retrieval passages based on config and metadata.

        Args:
            metadata: Image metadata containing patient info, modality, etc.
            visual_analysis: Optional visual analysis results for query building

        Returns:
            List of retrieved passages for context augmentation.
        """
        if not self.config.use_retrieval or self._retrieval_manager is None:
            return []

        return self._retrieval_manager.retrieve_for_task(
            task="localization",
            metadata=metadata,
            visual_analysis=visual_analysis,
        )

    def _convert_result(
        self,
        result: AgenticResult,
        image_path: Path,
        batch_idx: int,
        sample_idx: int,
    ) -> ModelResponse:
        """Convert AgenticResult to ModelResponse."""
        response = result.final_response

        # Extract boxes and labels
        boxes = response.get("boxes", [])
        labels = response.get("labels", [])
        reasoning = response.get("reasoning", "")

        # Add visual analysis context to reasoning if available
        if result.visual_analysis:
            va = result.visual_analysis
            reasoning += f"\n\nVisual pre-analysis: {va.overall_assessment}"
            reasoning += f" (symmetry: {va.symmetry_analysis.symmetry_score:.2f})"

        return ModelResponse(
            text=json.dumps({"boxes": boxes, "labels": labels}),
            confidence=result.confidence,
            reasoning=reasoning,
            metadata={
                "image_path": str(image_path),
                "num_boxes": len(boxes),
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
                "num_turns": len(result.turns),
                "total_tokens": result.total_tokens,
                "used_visual_reasoning": result.visual_analysis is not None,
                "tool_calls": sum(len(t.tool_calls) for t in result.turns),
            },
        )

    def _log_analysis(self, result: AgenticResult, sample_idx: int) -> None:
        """Log analysis details for debugging."""
        logger.info(
            f"Sample {sample_idx}: "
            f"confidence={result.confidence:.2f}, "
            f"turns={len(result.turns)}, "
            f"tokens={result.total_tokens}"
        )

        if result.visual_analysis:
            va = result.visual_analysis
            logger.debug(
                f"  Visual analysis: "
                f"symmetry={va.symmetry_analysis.symmetry_score:.2f}, "
                f"features={len(va.visual_features)}"
            )

    @beartype
    def evaluate_responses(
        self, responses: list[ModelResponse], ground_truth: list[str]
    ) -> EvaluationMetrics:
        """Evaluate localization responses."""
        # Extract predicted boxes - format for evaluator
        predicted_dicts = []
        for response in responses:
            try:
                parsed = json.loads(response.text)
                boxes = parsed.get("boxes", [])
                predicted_dicts.append(
                    {
                        "boxes": boxes,
                        "scores": [response.confidence] * len(boxes),
                        "labels": [0] * len(boxes),
                    }
                )
            except json.JSONDecodeError:
                predicted_dicts.append({"boxes": [], "scores": [], "labels": []})

        # Parse ground truth boxes
        ground_truth_dicts = []
        for gt in ground_truth:
            gt_parsed = json.loads(gt)
            boxes = gt_parsed.get("boxes", [])

            ground_truth_dicts.append(
                {
                    "boxes": boxes,
                    "scores": [1.0] * len(boxes),
                    "labels": [0] * len(boxes),
                }
            )

        # Use evaluation function
        results = evaluate_detection(predicted_dicts, ground_truth_dicts)

        return EvaluationMetrics(
            accuracy=results.get("map50", 0.0),
            precision=results.get("precision"),
            recall=results.get("recall"),
            f1_score=results.get("f1_score"),
            auc_roc=results.get("map30"),
        )
