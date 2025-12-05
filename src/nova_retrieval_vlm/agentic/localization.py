"""Agentic localization processor.

Enhanced localization with tool calling and multi-turn refinement.
"""

from __future__ import annotations

import json
from pathlib import Path

from beartype import beartype
from loguru import logger

from nova_retrieval_vlm.agentic.processor import AgenticProcessor
from nova_retrieval_vlm.agentic.processor import AgenticResult
from nova_retrieval_vlm.evaluation.detection import evaluate_detection
from nova_retrieval_vlm.processors.base import BaseProcessor
from nova_retrieval_vlm.processors.base import ProcessorConfig
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import DetectionMetrics
from nova_retrieval_vlm.types import ModelResponse


class AgenticLocalizationProcessor(BaseProcessor):
    """Agentic processor for localization tasks.

    Extends base localization with:
    - Tool calling for interactive analysis (zoom, crop, contrast, flip, rotate)
    - Multi-turn refinement for uncertain cases
    - Retrieval augmentation support
    """

    @beartype
    def __init__(
        self,
        config: ProcessorConfig,
        use_tools: bool = True,
        max_turns: int = 10,
    ) -> None:
        """Initialize agentic localization processor.

        Args:
            config: ProcessorConfig with model_name, output_dir, etc.
            use_tools: Enable tool calling (zoom, crop, web search, etc.)
            max_turns: Max turns for multi-turn refinement (default: 10)
        """
        super().__init__(config)
        self.use_tools = use_tools
        self.max_turns = max_turns
        self.task_name = config.task_name

        self._agentic_processor = AgenticProcessor(
            model_name=config.model_name,
            use_tools=use_tools,
            use_web_search=False,  # Localization typically doesn't need web search
            max_turns=max_turns,
            reasoning_enabled=config.reasoning_enabled,
            reasoning_effort=config.reasoning_effort,
            enable_caching=config.enable_caching,
        )

        logger.info(
            f"Initialized AgenticLocalizationProcessor with {max_turns} max turns, "
            f"tools={'enabled' if use_tools else 'disabled'}"
        )

    @beartype
    async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
        """Process localization batch with fully agentic analysis."""
        responses = []

        for i, (image_path, metadata) in enumerate(zip(batch.images, batch.metadata, strict=True)):
            self.logger.debug(f"Processing image {i + 1}/{len(batch.images)}: {image_path}")

            # Run agentic analysis - model can search web independently
            result = await self._agentic_processor.analyze(
                image_path=Path(image_path),
                metadata=metadata,
            )

            # Convert to ModelResponse
            response = self._convert_result(result, image_path, batch_idx, i)
            responses.append(response)

            # Log analysis details
            self._log_analysis(result, i)

        return responses

    @beartype
    def _convert_result(
        self,
        result: AgenticResult,
        image_path: str | Path,
        batch_idx: int,
        sample_idx: int,
    ) -> ModelResponse:
        """Convert AgenticResult to ModelResponse.

        Raises:
            KeyError: If required fields are missing from response.
        """
        response = result.final_response

        # Validate required fields exist - fail fast on missing data
        if "caption" not in response:
            raise KeyError("Missing 'caption' in agentic response")
        if "diagnosis" not in response:
            raise KeyError("Missing 'diagnosis' in agentic response")
        if "localization" not in response:
            raise KeyError("Missing 'localization' in agentic response")

        caption = response["caption"]
        diagnosis = response["diagnosis"]
        localization = response["localization"]
        # Optional field - chain-of-thought reasoning from model, defaults to empty
        reasoning: str = response.get("reasoning", "")

        # Extract boxes and labels from localization
        boxes = []
        labels = []
        localizations_list = localization.get("localizations", [])
        for loc in localizations_list:
            if "bounding_box" in loc:
                boxes.append(loc["bounding_box"])
            if "finding" in loc:
                labels.append(loc["finding"])

        return ModelResponse(
            text=json.dumps(
                {
                    "caption": caption,
                    "diagnosis": diagnosis,
                    "localization": localization,
                    "boxes": boxes,
                    "labels": labels,
                }
            ),
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

    @beartype
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
    ) -> DetectionMetrics:
        """Evaluate localization responses using NOVA detection metrics.

        Raises:
            ValueError: If JSON parsing fails or required fields are missing.
        """
        # Extract predicted boxes - format for evaluator
        predicted_dicts = []
        for response in responses:
            try:
                parsed = json.loads(response.text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in response: {response.text[:100]}") from e

            if "boxes" not in parsed:
                raise ValueError(f"Missing 'boxes' in response: {parsed}")
            boxes = parsed["boxes"]
            predicted_dicts.append(
                {
                    "boxes": boxes,
                    "scores": [response.confidence] * len(boxes),
                    "labels": [0] * len(boxes),
                }
            )

        # Parse ground truth boxes
        ground_truth_dicts = []
        for gt in ground_truth:
            gt_parsed = json.loads(gt)
            if "boxes" not in gt_parsed:
                raise ValueError(f"Missing 'boxes' in ground truth: {gt[:100]}")
            boxes = gt_parsed["boxes"]

            ground_truth_dicts.append(
                {
                    "boxes": boxes,
                    "scores": [1.0] * len(boxes),
                    "labels": [0] * len(boxes),
                }
            )

        # Use evaluation function
        results = evaluate_detection(predicted_dicts, ground_truth_dicts)

        return DetectionMetrics(
            map30=results["map30"],
            map50=results["map50"],
            map50_95=results["map50_95"],
            acc50=results["acc50"],
            tp30=results["tp30"],
            fp30=results["fp30"],
        )
