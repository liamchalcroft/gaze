"""Tests for Patch Set 3: Prompt safety, keyword F1 tokenization, and detection metrics."""

from __future__ import annotations

import pathlib
import re

import pytest

try:
    from examples.nova.src.evaluation.detection import IOU_THRESHOLD_LOOSE
    from examples.nova.src.evaluation.detection import IOU_THRESHOLD_STANDARD
    from examples.nova.src.evaluation.detection import clamp_and_validate_box
    from examples.nova.src.evaluation.detection import rescale_and_clamp_box

    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Keyword tokenization helper
# ---------------------------------------------------------------------------
# The caption module imports through evaluation/__init__.py which pulls in
# detection.py → torch. We replicate the helper here to test independently.

_WORD_PATTERN = re.compile(r"\b[\w][\w-]*[\w]\b|\b\w\b")


def _extract_keyword_tokens(text: str) -> set[str]:
    """Local copy of the helper from caption.py for testing without torch."""
    tokens: set[str] = set()
    for match in _WORD_PATTERN.finditer(text.lower()):
        word = match.group()
        tokens.add(word)
        if "-" in word:
            tokens.update(word.split("-"))
    return tokens


class TestExtractKeywordTokens:
    """Tests for _extract_keyword_tokens helper in caption evaluation."""

    def test_simple_words(self) -> None:
        result = _extract_keyword_tokens("axial T2 FLAIR")
        assert "axial" in result
        assert "t2" in result
        assert "flair" in result

    def test_hyphenated_compound_splits(self) -> None:
        """T2-weighted should yield both the compound and its parts."""
        result = _extract_keyword_tokens("T2-weighted")
        assert "t2-weighted" in result
        assert "t2" in result
        assert "weighted" in result

    def test_multiple_hyphenated(self) -> None:
        result = _extract_keyword_tokens("T1-weighted axial FLAIR T2-weighted sagittal")
        # Compounds
        assert "t1-weighted" in result
        assert "t2-weighted" in result
        # Parts
        assert "t1" in result
        assert "t2" in result
        assert "weighted" in result
        # Plain words
        assert "axial" in result
        assert "flair" in result
        assert "sagittal" in result

    def test_case_insensitive(self) -> None:
        result = _extract_keyword_tokens("FLAIR Axial DWI")
        assert "flair" in result
        assert "axial" in result
        assert "dwi" in result

    def test_punctuation_stripped(self) -> None:
        """Punctuation like commas and periods should not be part of tokens."""
        result = _extract_keyword_tokens("lesion, tumor. cyst!")
        assert "lesion" in result
        assert "tumor" in result
        assert "cyst" in result
        # Should NOT contain punctuation-bearing variants
        assert "lesion," not in result
        assert "tumor." not in result

    def test_empty_string(self) -> None:
        result = _extract_keyword_tokens("")
        assert result == set()

    def test_single_char_words(self) -> None:
        """Single character words should be captured by the regex."""
        result = _extract_keyword_tokens("a T2 scan")
        assert "a" in result
        assert "t2" in result
        assert "scan" in result

    def test_clinical_term_matching(self) -> None:
        """Verify that clinical terms from NOVA evaluation are matched."""
        text = "lesion with hemorrhage and edema causing mass effect"
        result = _extract_keyword_tokens(text)
        for term in ["lesion", "hemorrhage", "edema", "mass"]:
            assert term in result, f"Expected '{term}' in tokens"

    def test_modality_term_matching(self) -> None:
        """Verify modality terms from NOVA evaluation are matched, including via hyphen splitting."""
        text = "Axial T2-weighted FLAIR showing coronal DWI"
        result = _extract_keyword_tokens(text)
        expected = {"flair", "t2", "axial", "coronal", "dwi", "weighted"}
        for term in expected:
            assert term in result, f"Expected modality term '{term}' in tokens"


class TestKeywordTokenSourceParity:
    """Verify our local copy matches the real source code."""

    def test_source_contains_identical_regex(self) -> None:
        """The regex and helper in caption.py should match this test's copy."""
        source = (
            pathlib.Path(__file__).resolve().parents[1]
            / "examples"
            / "nova"
            / "src"
            / "evaluation"
            / "caption.py"
        ).read_text()
        assert r"\b[\w][\w-]*[\w]\b|\b\w\b" in source
        assert "def _extract_keyword_tokens(text: str)" in source
        assert 'tokens.update(word.split("-"))' in source


# ---------------------------------------------------------------------------
# 2. Prompt disclaimer and bias checks
# ---------------------------------------------------------------------------

PROMPT_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples" / "nova" / "src" / "prompts"


class TestPromptDisclaimer:
    """Verify research disclaimer is present in system prompts."""

    def test_single_turn_system_has_disclaimer(self) -> None:
        text = (PROMPT_DIR / "single_turn" / "system.jinja").read_text()
        assert "research benchmark evaluation" in text.lower()
        assert "not a clinical decision-support system" in text.lower()

    def test_agentic_system_has_disclaimer(self) -> None:
        text = (PROMPT_DIR / "agentic" / "system.jinja").read_text()
        assert "research benchmark evaluation" in text.lower()
        assert "not a clinical decision-support system" in text.lower()


class TestPathologyBiasSoftened:
    """Verify prompts use softer language about confirmed pathology."""

    def test_single_turn_task_no_shouting(self) -> None:
        """Task prompt should not use all-caps emphatic statements about pathology."""
        text = (PROMPT_DIR / "single_turn" / "task.jinja").read_text()
        assert "EVERY CASE" not in text
        assert "CONFIRMED PATHOLOGY" not in text
        # Should use softer phrasing
        assert "confirmed abnormalities" in text or "confirmed pathology" in text

    def test_single_turn_task_no_clinical_decision_claim(self) -> None:
        """Should not claim analysis guides clinical decisions."""
        text = (PROMPT_DIR / "single_turn" / "task.jinja").read_text()
        assert "guide clinical decisions" not in text.lower()

    def test_agentic_task_no_shouting(self) -> None:
        text = (PROMPT_DIR / "agentic" / "task.jinja").read_text()
        assert "EVERY CASE" not in text

    def test_agentic_system_benchmark_context(self) -> None:
        """Agentic system prompt should frame as benchmark, not clinical."""
        text = (PROMPT_DIR / "agentic" / "system.jinja").read_text()
        assert "benchmark" in text.lower()


# ---------------------------------------------------------------------------
# 3. Detection metrics: per-axis scaling and IoU threshold documentation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DETECTION_AVAILABLE, reason="detection module not available")
class TestPerAxisScalingSpatialAccuracy:
    """Per-axis scaling must preserve position on non-overflowing axes.

    This is critical for medical imaging: if a model correctly localizes a
    lesion on the y-axis but outputs x-coords in a larger coordinate space,
    uniform scaling would shift the y-position and potentially miss the lesion.
    """

    def test_x_overflow_preserves_y_position(self) -> None:
        """Y-coordinates must not change when only x overflows."""
        result = rescale_and_clamp_box([200, 100, 800, 200], 480, 480)
        # scale_x = 480/800 = 0.6; scale_y = 1.0 (200 < 480)
        assert result[1] == 100.0, "y1 must not change when y is within bounds"
        assert result[3] == 200.0, "y2 must not change when y is within bounds"

    def test_y_overflow_preserves_x_position(self) -> None:
        """X-coordinates must not change when only y overflows."""
        result = rescale_and_clamp_box([100, 200, 200, 800], 480, 480)
        # scale_x = 1.0 (200 < 480); scale_y = 480/800 = 0.6
        assert result[0] == 100.0, "x1 must not change when x is within bounds"
        assert result[2] == 200.0, "x2 must not change when x is within bounds"

    def test_both_overflow_independent_scales(self) -> None:
        """Each axis must be scaled by its own factor."""
        # Model thinks image is 960x720, actual is 480x480
        # scale_x = 480/960 = 0.5, scale_y = 480/720 = 2/3
        result = rescale_and_clamp_box([96, 72, 960, 720], 480, 480)
        assert abs(result[0] - 48.0) < 1e-6  # 96 * 0.5
        assert abs(result[1] - 48.0) < 1e-6  # 72 * (2/3)
        assert abs(result[2] - 480.0) < 1e-6  # 960 * 0.5
        assert abs(result[3] - 480.0) < 1e-6  # 720 * (2/3)

    def test_no_overflow_passthrough(self) -> None:
        """In-bounds box must pass through unchanged."""
        box = [50, 60, 200, 300]
        assert rescale_and_clamp_box(box, 480, 480) == [50.0, 60.0, 200.0, 300.0]

    def test_exact_boundary_no_rescale(self) -> None:
        """Box exactly at image boundary should not be rescaled."""
        result = rescale_and_clamp_box([0, 0, 480, 480], 480, 480)
        assert result == [0.0, 0.0, 480.0, 480.0]


@pytest.mark.skipif(not DETECTION_AVAILABLE, reason="detection module not available")
class TestRescaleEdgeCases:
    """Edge cases for rescale_and_clamp_box."""

    def test_single_pixel_box_overflow(self) -> None:
        """Degenerate single-pixel box outside bounds."""
        result = rescale_and_clamp_box([960, 960, 960, 960], 480, 480)
        assert result == [480.0, 480.0, 480.0, 480.0]

    def test_swapped_coordinates_then_rescale(self) -> None:
        """Swapped coords get fixed before rescaling."""
        result = rescale_and_clamp_box([960, 720, 0, 0], 480, 480)
        assert result[0] <= result[2]
        assert result[1] <= result[3]
        # After swap: [0, 0, 960, 720]
        # scale_x = 0.5, scale_y = 2/3 → [0, 0, 480, 480]
        assert abs(result[2] - 480.0) < 1e-6
        assert abs(result[3] - 480.0) < 1e-6

    def test_fractional_coordinates(self) -> None:
        """Fractional coordinates should be handled correctly."""
        result = rescale_and_clamp_box([0.5, 0.5, 960.5, 240.5], 480, 480)
        # max_x = 960.5 > 480 → scale_x = 480/960.5
        # max_y = 240.5 < 480 → scale_y = 1.0
        assert result[1] == 0.5  # y1 unchanged
        assert result[3] == 240.5  # y2 unchanged
        assert result[2] <= 480.0  # x2 within bounds

    def test_negative_coordinates_clamped(self) -> None:
        """Negative coordinates should be clamped to 0."""
        result = rescale_and_clamp_box([-10, -20, 100, 200], 480, 480)
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_non_square_image(self) -> None:
        """Per-axis scaling on a non-square image."""
        # Image is 640x480, box is [0, 0, 1280, 960]
        # scale_x = 640/1280 = 0.5, scale_y = 480/960 = 0.5
        result = rescale_and_clamp_box([0, 0, 1280, 960], 640, 480)
        assert result == [0.0, 0.0, 640.0, 480.0]

    def test_non_square_image_single_axis(self) -> None:
        """Non-square image, only x overflows."""
        # Image is 640x480, box is [0, 0, 1280, 200]
        # scale_x = 640/1280 = 0.5, scale_y = 1.0 (200 < 480)
        result = rescale_and_clamp_box([0, 0, 1280, 200], 640, 480)
        assert result == [0.0, 0.0, 640.0, 200.0]

    def test_non_square_coord_space_mapping(self) -> None:
        """Model thinks image is 1000x500 when it is actually 480x480."""
        # Box in model's space: [100, 50, 800, 400]
        # max_x = 800 > 480 → scale_x = 480/800 = 0.6
        # max_y = 400 < 480 → scale_y = 1.0
        result = rescale_and_clamp_box([100, 50, 800, 400], 480, 480)
        assert abs(result[0] - 60.0) < 1e-6  # 100 * 0.6
        assert result[1] == 50.0  # unchanged
        assert abs(result[2] - 480.0) < 1e-6  # 800 * 0.6
        assert result[3] == 400.0  # unchanged


@pytest.mark.skipif(not DETECTION_AVAILABLE, reason="detection module not available")
class TestClampAndValidateBox:
    """Ensure clamp_and_validate_box still works correctly (no changes)."""

    def test_basic_clamp(self) -> None:
        result = clamp_and_validate_box([0, 0, 100, 100], 480, 480)
        assert result == [0.0, 0.0, 100.0, 100.0]

    def test_clamp_overflow(self) -> None:
        result = clamp_and_validate_box([0, 0, 600, 700], 480, 480)
        assert result == [0.0, 0.0, 480.0, 480.0]

    def test_clamp_negative(self) -> None:
        result = clamp_and_validate_box([-10, -20, 100, 200], 480, 480)
        assert result == [0.0, 0.0, 100.0, 200.0]

    def test_swapped(self) -> None:
        result = clamp_and_validate_box([200, 300, 100, 50], 480, 480)
        assert result[0] <= result[2]
        assert result[1] <= result[3]


@pytest.mark.skipif(not DETECTION_AVAILABLE, reason="detection module not available")
class TestIoUThresholdConstants:
    """Verify IoU threshold constants match NOVA protocol."""

    def test_loose_threshold_value(self) -> None:
        assert IOU_THRESHOLD_LOOSE == 0.3

    def test_standard_threshold_value(self) -> None:
        assert IOU_THRESHOLD_STANDARD == 0.5

    def test_standard_more_strict_than_loose(self) -> None:
        assert IOU_THRESHOLD_STANDARD > IOU_THRESHOLD_LOOSE
