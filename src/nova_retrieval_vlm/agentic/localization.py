"""Agentic localization processor.

Enhanced localization with tool calling and multi-turn refinement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger

from nova_retrieval_vlm.agentic.processor import AgenticProcessor
from nova_retrieval_vlm.agentic.processor import AgenticResult
from nova_retrieval_vlm.evaluation.detection import evaluate_detection
from nova_retrieval_vlm.processors.base import BaseProcessor
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import ModelResponse


class AgenticLocalizationProcessor(BaseProcessor):
    """Agentic processor for localization tasks.

    Extends base localization with:
    - Tool calling for interactive analysis (zoom, crop, contrast, flip, rotate)
    - Multi-turn refinement for uncertain cases
    - Retrieval augmentation support
    - Visual comparison with retrieved examples (planned)
    """

    def __init__(
        self,
        config: Any,
        use_tools: bool = True,
        max_turns: int = 10,
    ):
        """Initialize agentic localization processor.

        Args:
            config: ProcessorConfig with model_name, output_dir, etc.
            use_tools: Enable tool calling (zoom, crop, web search, etc.)
            max_turns: Max turns for multi-turn refinement (default: 10)
        """
        super().__init__(config)
        self.use_tools = use_tools
        self.max_turns = max_turns

        self._agentic_processor = AgenticProcessor(
            model_name=config.model_name,
            use_tools=use_tools,
            max_turns=max_turns,
        )

        logger.info(
            f"Initialized AgenticLocalizationProcessor with {max_turns} max turns, "
            f"tools={'enabled' if use_tools else 'disabled'}"
        )

    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process localization batch with fully agentic analysis."""
        responses = []

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=False)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Run agentic analysis - model can search web independently
            result = await self._agentic_processor.analyze(
                image_path=Path(image_path),
                task="localization",
                metadata=metadata,
            )

            # Convert to ModelResponse
            response = self._convert_result(result, image_path, batch_idx, i)
            responses.append(response)

            # Log analysis details
            self._log_analysis(result, i)

        return responses

    # NOTE: Retrieval method removed - model now uses search_web tool for independent information retrieval

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
                "tool_calls": sum(len(t.tool_calls) for t in result.turns),
                "web_searches_used": sum(
                    1
                    for t in result.turns
                    if any(tc.get("name") == "search_web" for tc in t.tool_calls)
                ),
            },
        )

    def _log_analysis(self, result: AgenticResult, sample_idx: int) -> None:
        """Log analysis details for debugging."""
        tool_count = sum(len(t.tool_calls) for t in result.turns)
        logger.info(
            f"Sample {sample_idx}: "
            f"confidence={result.confidence:.2f}, "
            f"turns={len(result.turns)}, "
            f"tools={tool_count}, "
            f"tokens={result.total_tokens}"
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
