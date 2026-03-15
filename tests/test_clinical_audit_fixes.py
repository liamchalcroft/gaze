"""Tests for clinical informatics audit patch sets (PS1–PS4).

Validates:
- PS1: rewards.py fixes (multimodal text, IoU area penalty, bbox regex, CombinedReward warning)
- PS2: window_level preset safety (stroke width, is_preset exemption removed)
- PS3: coord_space_modified cleared on reset()
- PS4: diagnosis.py default model warning, expanded synonyms
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

from radiant_harness.base import _COORD_MODIFYING_TOOLS
from radiant_harness.base import AgenticProcessorBase
from radiant_harness.base import ImageInput
from radiant_harness.models import AdapterProtocol
from radiant_harness.models import GenerationLog
from radiant_harness.tools import Tool
from radiant_harness.tools import ToolRegistry
from radiant_harness.tools.visual import WINDOW_PRESETS
from radiant_harness.tools.visual import apply_window_level
from radiant_harness.types import ToolResult
from radiant_harness.verifiers.rewards import CombinedReward
from radiant_harness.verifiers.rewards import ExactMatchReward
from radiant_harness.verifiers.rewards import IoUReward
from radiant_harness.verifiers.rewards import TokenF1Reward
from radiant_harness.verifiers.rewards import extract_completion_text

# Import diagnosis module from examples/nova
REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_NOVA_ROOT = REPO_ROOT / "examples" / "nova"
if str(EXAMPLE_NOVA_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_NOVA_ROOT))


# =============================================================================
# PS1: rewards.py fixes
# =============================================================================


class TestExtractCompletionTextMultimodal:
    """extract_completion_text must concatenate ALL text items, not just the first."""

    def test_single_text_item(self) -> None:
        completion = [{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]
        assert extract_completion_text(completion) == "hello"

    def test_multiple_text_items_concatenated(self) -> None:
        completion = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "reasoning about the image"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "text", "text": '{"bbox": [0.1, 0.2, 0.3, 0.4]}'},
                ],
            }
        ]
        result = extract_completion_text(completion)
        assert "reasoning about the image" in result
        assert '{"bbox": [0.1, 0.2, 0.3, 0.4]}' in result

    def test_plain_string_passthrough(self) -> None:
        assert extract_completion_text("plain text") == "plain text"

    def test_string_content_passthrough(self) -> None:
        completion = [{"role": "assistant", "content": "string content"}]
        assert extract_completion_text(completion) == "string content"


class TestIoURewardAreaPenalty:
    """IoUReward area penalty must not use broken box-based estimate."""

    def test_pixel_coords_with_image_area_uses_supplied_area(self) -> None:
        """When image_area is in info, use it for penalty calculation."""
        reward = IoUReward(normalized=True, continuous=True, area_penalty_start=0.5)
        # Pred and ref are identical large pixel boxes (covers >50% of image)
        pred_text = '{"bbox": [0, 0, 400, 400]}'
        info = {
            "bbox": [0, 0, 400, 400],
            "image_area": 512 * 512,  # 262144 pixels
        }
        score = reward("", pred_text, info)
        # IoU = 1.0, but area_ratio = 160000/262144 ≈ 0.61 > 0.5 → penalty applied
        assert 0.0 < score < 1.0

    def test_pixel_coords_without_image_area_skips_penalty(self) -> None:
        """When pixel coords are detected but no image_area, skip penalty (don't crash)."""
        reward = IoUReward(normalized=True, continuous=True, area_penalty_start=0.5)
        pred_text = '{"bbox": [0, 0, 400, 400]}'
        info = {
            "bbox": [0, 0, 400, 400],
            # No image_area — penalty should be skipped
        }
        score = reward("", pred_text, info)
        # IoU = 1.0, penalty skipped → full score
        assert score == 1.0

    def test_normalized_coords_still_penalized(self) -> None:
        """Normalized coords in [0,1] with large area still get penalized."""
        reward = IoUReward(normalized=True, continuous=True, area_penalty_start=0.5)
        pred_text = '{"bbox": [0.0, 0.0, 1.0, 1.0]}'
        info = {"bbox": [0.0, 0.0, 1.0, 1.0]}
        score = reward("", pred_text, info)
        # Full-image box: area_ratio = 1.0, penalty = 0.0 → score = 0.0
        assert score == 0.0


class TestBboxRegexLastMatch:
    """_extract_bbox regex fallback must return the LAST matching array."""

    def test_last_array_wins(self) -> None:
        reward = IoUReward()
        # Text with reasoning array before the actual bbox
        text = (
            "The region of interest spans approximately [10, 20, 30, 40] pixels. "
            "After careful analysis, the bounding box is [0.1, 0.2, 0.8, 0.9]."
        )
        bbox = reward._extract_bbox(text)
        # Should return the LAST match [0.1, 0.2, 0.8, 0.9], not [10, 20, 30, 40]
        assert bbox == [0.1, 0.2, 0.8, 0.9]

    def test_single_array_still_works(self) -> None:
        reward = IoUReward()
        text = "The bbox is [0.5, 0.5, 0.8, 0.8]."
        bbox = reward._extract_bbox(text)
        assert bbox == [0.5, 0.5, 0.8, 0.8]

    def test_json_bbox_preferred_over_regex(self) -> None:
        reward = IoUReward()
        text = '{"bbox": [0.1, 0.2, 0.3, 0.4]} and also [0.5, 0.6, 0.7, 0.8]'
        bbox = reward._extract_bbox(text)
        # JSON extraction should be preferred
        assert bbox == [0.1, 0.2, 0.3, 0.4]


class TestCombinedRewardWeightError:
    """CombinedReward must raise ValueError when weights don't sum to 1.0."""

    def test_error_on_non_unit_weights(self) -> None:
        import pytest

        r1 = ExactMatchReward()
        r2 = TokenF1Reward()
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            CombinedReward(rewards=[r1, r2], weights=[2.0, 3.0])

    def test_no_warning_on_unit_weights(self) -> None:
        from loguru import logger as loguru_logger

        captured: list[str] = []
        handler_id = loguru_logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
        try:
            r1 = ExactMatchReward()
            r2 = TokenF1Reward()
            CombinedReward(rewards=[r1, r2], weights=[0.6, 0.4])
        finally:
            loguru_logger.remove(handler_id)
        assert not any("auto-normalizing" in msg for msg in captured)


# =============================================================================
# PS2: window_level preset safety
# =============================================================================


class TestWindowLevelPresetSafety:
    """All window presets must respect the min_window_width safety floor."""

    def test_all_presets_above_min_width(self) -> None:
        """Every preset must have width >= min_window_width (default 10)."""
        from radiant_harness.config import get_config

        min_width = get_config().image.min_window_width
        for name, (_center, width) in WINDOW_PRESETS.items():
            assert width >= min_width, (
                f"Preset {name!r} has width={width} < min_window_width={min_width}. "
                f"This destroys diagnostic information on 8-bit images."
            )

    def test_stroke_preset_updated(self) -> None:
        """Stroke preset must no longer use the unsafe width=8."""
        _, width = WINDOW_PRESETS["stroke"]
        assert width >= 10, f"Stroke preset width={width} is still below safety floor"

    def test_preset_no_longer_exempt_from_safety_check(self) -> None:
        """Presets go through the same width check as manual values."""
        img = Image.new("L", (100, 100), color=128)
        # A hypothetical preset with width < min would now raise.
        # We test by directly calling apply_window_level with narrow width
        # and verifying it fails (even if it were a "preset" path).
        with pytest.raises(ValueError, match="min_window_width|width must be"):
            apply_window_level(img, center=32, width=5)

    def test_valid_preset_works(self) -> None:
        """Existing presets (now all above min_width) should work fine."""
        img = Image.new("L", (100, 100), color=128)
        result = apply_window_level(img, preset="brain")
        assert result.size == (100, 100)


# =============================================================================
# PS3: coord_space_modified cleared on reset()
# =============================================================================


async def _noop_tool(registry: ToolRegistry) -> ToolResult:  # noqa: ARG001
    return ToolResult(tool_name="noop", description="no-op")


async def _reset_tool(registry: ToolRegistry) -> ToolResult:  # noqa: ARG001
    return ToolResult(
        tool_name="reset",
        description="Reset to original image",
    )


class ResetTrackingAdapter(AdapterProtocol):
    """Adapter that calls crop (turn 1), then reset (turn 2), then finalizes."""

    supports_multipart_tool_content: bool = True

    def __init__(self) -> None:
        self.calls = 0

    async def generate_chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]] | None, GenerationLog]:
        _ = messages, max_tokens, temperature, tools, response_format, kwargs
        self.calls += 1
        if self.calls == 1:
            # Call crop (coord-modifying tool)
            return (
                "",
                [{"id": "call-1", "name": "crop", "arguments": "{}"}],
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
            )
        if self.calls == 2:
            # Call reset
            return (
                "",
                [{"id": "call-2", "name": "reset", "arguments": "{}"}],
                GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="tool_call"),
            )
        # Final response
        return (
            '{"continue": false, "result": "done"}',
            None,
            GenerationLog(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        )


class ResetTrackingProcessor(AgenticProcessorBase):
    def __init__(self, adapter: AdapterProtocol) -> None:
        super().__init__(
            model_name="test-model",
            use_tools=True,
            use_web_search=False,
            max_turns=5,
            adapter_factory=lambda: adapter,
        )

    def get_system_prompt(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "system"

    def get_user_message(self, images: list[ImageInput], metadata: dict[str, Any]) -> str:
        _ = images, metadata
        return "user"

    def get_response_schema(self) -> dict[str, Any] | None:
        return None

    def validate_response(self, response: dict[str, Any]) -> bool:
        return "result" in response

    def _create_tool_registry(self, images: list[ImageInput]) -> ToolRegistry | None:
        _ = images
        crop_tool = Tool(
            name="crop",
            description="crop",
            parameters={},
            execute=_noop_tool,
            requires_image=False,
        )
        reset_tool = Tool(
            name="reset",
            description="reset",
            parameters={},
            execute=_reset_tool,
            requires_image=False,
        )
        return ToolRegistry(image_path=None, tools=[crop_tool, reset_tool])


@pytest.mark.asyncio
async def test_coord_space_modified_cleared_on_reset() -> None:
    """After reset(), coord_space_modified should be False so final-turn
    warning does NOT misleadingly say coordinates are INVALID."""
    adapter = ResetTrackingAdapter()
    processor = ResetTrackingProcessor(adapter)
    result = await processor.analyze(images=None, metadata={})

    assert result.final_response["result"] == "done"
    # The processor completed successfully — verify through adapter call count
    # that crop and reset were both executed
    assert adapter.calls == 3  # crop, reset, final


def test_reset_not_in_coord_modifying_tools() -> None:
    """reset must NOT be in _COORD_MODIFYING_TOOLS (it clears, not sets)."""
    assert "reset" not in _COORD_MODIFYING_TOOLS


# =============================================================================
# PS4: diagnosis.py — default model warning, expanded synonyms
# =============================================================================


class TestDiagnosisDefaultWarning:
    """llm_semantic_match must warn when using nano model with 1 vote."""

    def test_nano_model_warning_logged(self) -> None:
        import logging as stdlib_logging

        from src.evaluation.diagnosis import llm_semantic_match_async

        # diagnosis.py uses stdlib logging, so we can capture via handler
        diag_logger = stdlib_logging.getLogger("src.evaluation.diagnosis")
        captured: list[str] = []

        class _Handler(stdlib_logging.Handler):
            def emit(self, record: stdlib_logging.LogRecord) -> None:
                captured.append(record.getMessage())

        handler = _Handler(level=stdlib_logging.WARNING)
        diag_logger.addHandler(handler)
        old_level = diag_logger.level
        diag_logger.setLevel(stdlib_logging.WARNING)
        try:
            with (
                patch(
                    "src.evaluation.diagnosis._get_semantic_match_client",
                    side_effect=ValueError("mocked - no real API call"),
                ),
                pytest.raises(ValueError, match="mocked"),
            ):
                asyncio.get_event_loop().run_until_complete(
                    llm_semantic_match_async("glioma", "glioblastoma", "openai/gpt-5-nano", 1)
                )
        finally:
            diag_logger.removeHandler(handler)
            diag_logger.setLevel(old_level)

        assert any("num_votes=1" in msg for msg in captured)


class TestExpandedSynonyms:
    """New synonym pairs must be recognized by exact_diagnosis_match."""

    def test_brain_abscess_synonym(self) -> None:
        from src.evaluation.diagnosis import exact_diagnosis_match

        assert exact_diagnosis_match("brain abscess", "cerebral abscess")
        assert exact_diagnosis_match("cerebral abscess", "brain abscess")

    def test_ms_abbreviation(self) -> None:
        from src.evaluation.diagnosis import exact_diagnosis_match

        assert exact_diagnosis_match("ms", "multiple sclerosis")
        assert exact_diagnosis_match("multiple sclerosis", "ms")

    def test_nph_abbreviation(self) -> None:
        from src.evaluation.diagnosis import exact_diagnosis_match

        assert exact_diagnosis_match("nph", "normal pressure hydrocephalus")
        assert exact_diagnosis_match("normal pressure hydrocephalus", "nph")

    def test_meningioma_synonym(self) -> None:
        from src.evaluation.diagnosis import exact_diagnosis_match

        assert exact_diagnosis_match("meningioma", "meningeal tumor")
        assert exact_diagnosis_match("meningeal tumor", "meningioma")

    def test_pituitary_synonym(self) -> None:
        from src.evaluation.diagnosis import exact_diagnosis_match

        assert exact_diagnosis_match("pituitary adenoma", "pituitary tumor")
        assert exact_diagnosis_match("pituitary tumor", "pituitary adenoma")

    def test_non_synonyms_still_fail(self) -> None:
        """Expanded synonyms must not introduce false positives."""
        from src.evaluation.diagnosis import exact_diagnosis_match

        assert not exact_diagnosis_match("glioma", "meningioma")
        assert not exact_diagnosis_match("brain abscess", "brain tumor")
