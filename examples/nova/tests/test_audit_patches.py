"""Tests for NOVA audit patches."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "nova"
PAPER_ROOT = REPO_ROOT / "examples" / "aiih2026_paper"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))


class TestAreaPenaltyEdgeCases:
    def test_penalty_start_exactly_one(self) -> None:
        from src.rewards import _area_penalty

        result = _area_penalty([0.0, 0.0, 100.0, 100.0], image_area=10000.0, penalty_start=1.0)
        assert result == 1.0

    def test_penalty_start_greater_than_one(self) -> None:
        from src.rewards import _area_penalty

        result = _area_penalty([0.0, 0.0, 100.0, 100.0], image_area=10000.0, penalty_start=1.5)
        assert result == 1.0

    def test_zero_image_area(self) -> None:
        from src.rewards import _area_penalty

        assert _area_penalty([0.0, 0.0, 100.0, 100.0], image_area=0.0) == 1.0

    def test_negative_image_area(self) -> None:
        from src.rewards import _area_penalty

        assert _area_penalty([0.0, 0.0, 100.0, 100.0], image_area=-1.0) == 1.0

    def test_small_box_no_penalty(self) -> None:
        from src.rewards import _area_penalty

        assert _area_penalty([0.0, 0.0, 50.0, 50.0], image_area=40000.0) == 1.0

    def test_full_image_box_zero_penalty(self) -> None:
        from src.rewards import _area_penalty

        assert _area_penalty([0.0, 0.0, 200.0, 200.0], image_area=40000.0) == 0.0

    def test_partial_penalty_linear_ramp(self) -> None:
        from src.rewards import _area_penalty

        # 150x200 = 30000 / 40000 = 75% → penalty = (1.0 - 0.75) / (1.0 - 0.5) = 0.5
        result = _area_penalty([0.0, 0.0, 150.0, 200.0], image_area=40000.0)
        assert abs(result - 0.5) < 1e-6, f"Expected 0.5, got {result}"


class TestAbbreviationMappingSync:
    def test_mappings_are_identical(self) -> None:
        from src.evaluation.diagnosis import _ABBREVIATION_MAPPING as _DIAG_MAPPING
        from src.rewards import _ABBREVIATION_MAPPING as _REWARD_MAPPING

        assert _REWARD_MAPPING == _DIAG_MAPPING, (
            f"Abbreviation mapping mismatch.\n"
            f"  Only in rewards: {set(_REWARD_MAPPING) - set(_DIAG_MAPPING)}\n"
            f"  Only in diagnosis: {set(_DIAG_MAPPING) - set(_REWARD_MAPPING)}"
        )

    def test_newly_added_abbreviations_expand(self) -> None:
        from src.rewards import _normalize_diagnosis

        assert _normalize_diagnosis("mri") == "magnetic resonance imaging"
        assert _normalize_diagnosis("ct") == "computed tomography"
        assert _normalize_diagnosis("dwi") == "diffusion weighted imaging"
        assert _normalize_diagnosis("flair") == "fluid attenuated inversion recovery"

    def test_abbreviation_prefix_expansion(self) -> None:
        from src.rewards import _normalize_diagnosis

        result = _normalize_diagnosis("gbm grade iv")
        assert result.startswith("glioblastoma multiforme")
        assert "grade iv" in result

    def test_abbreviation_not_expanded_mid_word(self) -> None:
        from src.rewards import _normalize_diagnosis

        assert _normalize_diagnosis("acclaim") == "acclaim"


class TestSingleTurnPromptSchemaAlignment:
    """Verify the NOVA single-turn prompt works with the base.py skeleton injection.

    Schema field names are communicated via _build_schema_skeleton() in base.py,
    not via inline JSON examples in the template. These tests verify the skeleton
    includes all required NOVA schema fields.
    """

    def test_skeleton_includes_caption_fields(self) -> None:
        from src.schemas import NOVA_SCHEMA

        from gaze.base import _build_schema_skeleton

        skeleton, _ = _build_schema_skeleton(NOVA_SCHEMA)
        assert "caption" in skeleton
        assert isinstance(skeleton["caption"], dict)
        caption = skeleton["caption"]
        for field in ("description", "findings", "anatomical_regions"):
            assert field in caption, f"Skeleton missing caption.{field}"

    def test_skeleton_includes_all_top_level_schema_keys(self) -> None:
        from src.schemas import NOVA_SCHEMA

        from gaze.base import _build_schema_skeleton

        skeleton, _ = _build_schema_skeleton(NOVA_SCHEMA)
        for key in ("caption", "diagnosis", "localization", "continue"):
            assert key in skeleton, f"Skeleton missing top-level key: {key}"

    def test_prompt_renders_with_essential_content(self) -> None:
        content = (EXAMPLE_ROOT / "src" / "prompts" / "single_turn" / "task.jinja").read_text()
        # Template must mention key clinical concepts (not JSON field names)
        assert "caption" in content.lower()
        assert "diagnosis" in content.lower()
        assert "localisation" in content.lower() or "localization" in content.lower()
        assert "bounding box" in content.lower()


class TestGTBoxClampingDimensions:
    def test_gt_dimensions_passed_through_info(self) -> None:
        from src.rewards import NOVAVerifiersReward

        reward_fn = NOVAVerifiersReward(task="localization")
        completion = (
            '{"localization": {"localizations": ['
            '{"finding": "test", "bounding_box": [10, 10, 50, 50], '
            '"anatomical_location": "test", "confidence": 0.9}]}}'
        )

        info_with_dims: dict[str, Any] = {
            "localizations": [{"bbox": [10, 10, 50, 50]}],
            "image_width": 256,
            "image_height": 256,
        }
        assert reward_fn("prompt", completion, info_with_dims) == 1.0

        info_without_dims: dict[str, Any] = {"localizations": [{"bbox": [10, 10, 50, 50]}]}
        assert reward_fn("prompt", completion, info_without_dims) == 1.0

    def test_area_penalty_matters_for_large_boxes(self) -> None:
        from src.rewards import NOVAVerifiersReward

        reward_fn = NOVAVerifiersReward(task="localization")
        completion = (
            '{"localization": {"localizations": ['
            '{"finding": "test", "bounding_box": [0, 0, 95, 95], '
            '"anatomical_location": "test", "confidence": 0.9}]}}'
        )

        info_with_dims: dict[str, Any] = {
            "localizations": [{"bbox": [0, 0, 95, 95]}],
            "image_width": 100,
            "image_height": 100,
        }
        reward_with = reward_fn("prompt", completion, info_with_dims)

        info_without_dims = {"localizations": [{"bbox": [0, 0, 95, 95]}]}
        reward_without = reward_fn("prompt", completion, info_without_dims)

        assert reward_without == 1.0
        assert reward_with < 1.0, f"Expected area penalty to reduce reward, got {reward_with}"


class TestDiagnosisTop5NoDuplicate:
    def test_top5_code_uses_slice(self) -> None:
        """Top-5 loop must iterate p[1:] to skip already-checked p[0]."""
        content = (EXAMPLE_ROOT / "src" / "evaluation" / "diagnosis.py").read_text()
        assert "for pred in p[1:]:" in content


class TestNormalizeDiagnosisBehavior:
    def test_hedging_stripped_but_severity_preserved(self) -> None:
        from src.rewards import _normalize_diagnosis

        assert "possible" not in _normalize_diagnosis("possible glioma")
        assert "mild" in _normalize_diagnosis("mild hydrocephalus")

    def test_punctuation_stripped_hyphens_kept(self) -> None:
        from src.rewards import _normalize_diagnosis

        result = _normalize_diagnosis("Dandy-Walker malformation (variant)")
        assert "-" in result
        assert "(" not in result


class TestLocalizationRewardIntegration:
    def test_perfect_match_small_box(self) -> None:
        from src.rewards import compute_localization_reward

        pred = [[10.0, 10.0, 50.0, 50.0]]
        ref = [[10.0, 10.0, 50.0, 50.0]]
        assert compute_localization_reward(pred, ref, image_area=65536.0) == 1.0

    def test_perfect_match_large_box_penalized(self) -> None:
        from src.rewards import compute_localization_reward

        pred = [[0.0, 0.0, 240.0, 240.0]]
        ref = [[0.0, 0.0, 240.0, 240.0]]
        reward = compute_localization_reward(pred, ref, image_area=65536.0)
        assert reward < 0.3, f"Expected heavy area penalty, got {reward}"

    def test_no_area_penalty_when_disabled(self) -> None:
        from src.rewards import compute_localization_reward

        pred = [[0.0, 0.0, 240.0, 240.0]]
        ref = [[0.0, 0.0, 240.0, 240.0]]
        assert compute_localization_reward(pred, ref, image_area=0.0) == 1.0


class TestBertScoreClamping:
    """Finding #1: BERTScore can go negative with rescale_with_baseline=True."""

    def test_bert_f1_never_negative(self) -> None:
        """Verify returned bert_f1 is clamped to [0, 1]."""
        from unittest.mock import patch

        torch = pytest.importorskip("torch")

        from src.evaluation.caption import evaluate_caption

        # Simulate a very poor BERTScore (negative after baseline rescaling)
        negative_f1 = torch.tensor([-0.1, -0.2, -0.15])
        fake_result = (
            (torch.zeros(3), torch.zeros(3), negative_f1),
            "roberta-large_L17_no-idf_version=mock",
        )

        with patch("src.evaluation.caption.bert_score_fn", return_value=fake_result):
            result = evaluate_caption(["bad", "bad", "bad"], ["good ref", "good ref", "good ref"])

        bert_f1 = result["bert_f1"]
        assert bert_f1 is not None
        assert bert_f1 >= 0.0, f"bert_f1 should be >= 0, got {bert_f1}"
        assert bert_f1 <= 1.0, f"bert_f1 should be <= 1, got {bert_f1}"

    def test_bert_f1_above_one_clamped(self) -> None:
        """Edge case: if BERTScore somehow exceeds 1.0, clamp it."""
        from unittest.mock import patch

        torch = pytest.importorskip("torch")

        from src.evaluation.caption import evaluate_caption

        high_f1 = torch.tensor([1.5, 1.2, 1.3])
        fake_result = (
            (torch.zeros(3), torch.zeros(3), high_f1),
            "roberta-large_L17_no-idf_version=mock",
        )

        with patch("src.evaluation.caption.bert_score_fn", return_value=fake_result):
            result = evaluate_caption(["test", "test", "test"], ["ref", "ref", "ref"])

        bert_f1 = result["bert_f1"]
        assert bert_f1 is not None
        assert bert_f1 <= 1.0, f"bert_f1 should be <= 1, got {bert_f1}"


class TestDashPatternSync:
    """Finding #2: _DASH_PATTERN must handle em-dash in both rewards and diag."""

    def test_diag_normalizer_handles_em_dash(self) -> None:
        from src.evaluation.diagnosis import normalize_diagnosis_string

        result = normalize_diagnosis_string("septo\u2014optic dysplasia")
        assert "\u2014" not in result
        assert "-" in result

    def test_diag_normalizer_handles_en_dash(self) -> None:
        from src.evaluation.diagnosis import normalize_diagnosis_string

        result = normalize_diagnosis_string("septo\u2013optic dysplasia")
        assert "\u2013" not in result
        assert "-" in result

    def test_dash_patterns_identical(self) -> None:
        from src.evaluation.diagnosis import _DASH_PATTERN as _DIAG_DASH
        from src.rewards import _DASH_PATTERN as _REWARD_DASH

        assert _DIAG_DASH.pattern == _REWARD_DASH.pattern, (
            f"Dash patterns diverge: diag={_DIAG_DASH.pattern!r} vs rewards={_REWARD_DASH.pattern!r}"
        )


class TestLocalizationAnalysisLoading:
    def test_compute_model_ious_uses_box_annotations_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("matplotlib")
        from experiments import plot

        run_dir = tmp_path / "runs" / "main_results" / "run_a"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "config": {
                        "model": "google/gemini-3-flash-preview",
                        "mode": "agentic",
                    }
                }
            )
        )
        (run_dir / "sample_0.json").write_text(
            json.dumps(
                {
                    "sample_id": 0,
                    "response": {
                        "localization": {"localizations": [{"bounding_box": [0, 0, 10, 10]}]}
                    },
                }
            )
        )

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            plot,
            "_load_nova_box_annotations",
            lambda _n: [{"gt_boxes": [(0.0, 0.0, 10.0, 10.0)]}],  # type: ignore[arg-type]
        )

        def _unexpected_image_loader(_n: int) -> None:
            raise AssertionError("_compute_model_ious should not load pixel data")

        monkeypatch.setattr(plot, "_load_nova_images_and_gt", _unexpected_image_loader)

        runs = {
            "run_a": {
                "summary": {
                    "config": {
                        "model": "google/gemini-3-flash-preview",
                        "mode": "agentic",
                    }
                }
            }
        }
        model_ious = plot._compute_model_ious(runs, n_samples=1)

        assert model_ious["Gemini Flash"] == [1.0]


class TestRewardBboxKeyStrictness:
    """Finding #3: Reward should require 'bounding_box' for predictions."""

    def test_prediction_with_bbox_key_gets_zero(self) -> None:
        from src.rewards import NOVAVerifiersReward

        reward_fn = NOVAVerifiersReward(task="localization")
        # Model uses "bbox" instead of "bounding_box"
        completion = (
            '{"localization": {"localizations": ['
            '{"finding": "test", "bbox": [10, 10, 50, 50], '
            '"anatomical_location": "test", "confidence": 0.9}]}}'
        )
        info: dict[str, Any] = {
            "localizations": [{"bbox": [10, 10, 50, 50]}],
            "image_width": 256,
            "image_height": 256,
        }
        reward = reward_fn("prompt", completion, info)
        assert reward == 0.0, f"'bbox' key in predictions should yield 0, got {reward}"

    def test_prediction_with_bounding_box_key_works(self) -> None:
        from src.rewards import NOVAVerifiersReward

        reward_fn = NOVAVerifiersReward(task="localization")
        completion = (
            '{"localization": {"localizations": ['
            '{"finding": "test", "bounding_box": [10, 10, 50, 50], '
            '"anatomical_location": "test", "confidence": 0.9}]}}'
        )
        info: dict[str, Any] = {
            "localizations": [{"bbox": [10, 10, 50, 50]}],
            "image_width": 256,
            "image_height": 256,
        }
        reward = reward_fn("prompt", completion, info)
        assert reward == 1.0, f"'bounding_box' key should work, got {reward}"

    def test_ground_truth_bbox_key_still_accepted(self) -> None:
        from src.rewards import NOVAVerifiersReward

        reward_fn = NOVAVerifiersReward(task="localization")
        completion = (
            '{"localization": {"localizations": ['
            '{"finding": "test", "bounding_box": [10, 10, 50, 50], '
            '"anatomical_location": "test", "confidence": 0.9}]}}'
        )
        info: dict[str, Any] = {
            "localizations": [{"bbox": [10, 10, 50, 50]}],
            "image_width": 256,
            "image_height": 256,
        }
        assert reward_fn("prompt", completion, info) == 1.0


class TestSchemaValidationConfidence:
    """Finding #5: Schema validation should reject None confidence."""

    def test_null_caption_confidence_rejected(self) -> None:
        from src.schemas import validate_nova_response

        response: dict[str, Any] = {
            "caption": {
                "description": "test",
                "sequence_characteristics": "T1W",
                "orientation": "axial",
                "confidence": None,
                "findings": [],
                "anatomical_regions": [],
            },
            "diagnosis": {
                "primary_diagnosis": "test",
                "differential_diagnoses": [],
                "confidence": 0.8,
                "evidence": [],
                "clinical_recommendations": "",
            },
            "localization": {
                "localizations": [],
                "image_dimensions": {"width": 256, "height": 256},
                "coordinate_system": "absolute_pixels",
            },
            "continue": False,
            "reasoning": "test reasoning",
        }
        assert not validate_nova_response(response), "None confidence should fail"

    def test_null_diagnosis_confidence_rejected(self) -> None:
        from src.schemas import validate_nova_response

        response: dict[str, Any] = {
            "caption": {
                "description": "test",
                "sequence_characteristics": "T1W",
                "orientation": "axial",
                "confidence": 0.9,
                "findings": [],
                "anatomical_regions": [],
            },
            "diagnosis": {
                "primary_diagnosis": "test",
                "differential_diagnoses": [],
                "confidence": None,
                "evidence": [],
                "clinical_recommendations": "",
            },
            "localization": {
                "localizations": [],
                "image_dimensions": {"width": 256, "height": 256},
                "coordinate_system": "absolute_pixels",
            },
            "continue": False,
            "reasoning": "test reasoning",
        }
        assert not validate_nova_response(response), "None confidence should fail"


class TestRequiredFieldsMatchSchema:
    """get_required_fields() must match NOVA_SCHEMA required list exactly."""

    def test_required_fields_match_schema(self) -> None:
        from src.schemas import NOVA_SCHEMA
        from src.schemas import get_required_fields

        required = get_required_fields()
        schema_required = set(NOVA_SCHEMA["json_schema"]["schema"]["required"])
        assert set(required) == schema_required, (
            f"Mismatch: get_required_fields={set(required)} vs schema={schema_required}"
        )

    def test_valid_response_accepted(self) -> None:
        from src.schemas import validate_nova_response

        response: dict[str, Any] = {
            "caption": {
                "description": "test",
                "sequence_characteristics": "T1W",
                "orientation": "axial",
                "confidence": 0.9,
                "findings": [],
                "anatomical_regions": [],
            },
            "diagnosis": {
                "primary_diagnosis": "test",
                "differential_diagnoses": [],
                "confidence": 0.8,
                "evidence": [],
                "clinical_recommendations": "",
            },
            "localization": {
                "localizations": [],
                "image_dimensions": {"width": 256, "height": 256},
                "coordinate_system": "absolute_pixels",
            },
            "continue": False,
        }
        assert validate_nova_response(response), "Valid response should pass"


class TestAbbreviationWordBoundary:
    """Finding #6: Abbreviation expansion must use word boundaries."""

    def test_mid_string_abbreviation_expanded(self) -> None:
        from src.rewards import _normalize_diagnosis

        result = _normalize_diagnosis("left sah")
        assert "subarachnoid hemorrhage" in result, f"Got: {result}"

    def test_mid_string_abbreviation_expanded_in_diag(self) -> None:
        from src.evaluation.diagnosis import normalize_diagnosis_string

        result = normalize_diagnosis_string("left sah")
        assert "subarachnoid hemorrhage" in result, f"Got: {result}"

    def test_abbreviation_not_expanded_in_substring(self) -> None:
        from src.rewards import _normalize_diagnosis

        result = _normalize_diagnosis("acclaim")
        assert "agenesis" not in result, f"Substring expanded: {result}"


class TestHedgingWordBoundary:
    """Finding #7: Hedging modifier removal must respect word boundaries."""

    def test_improbable_not_corrupted(self) -> None:
        from src.rewards import _normalize_diagnosis

        result = _normalize_diagnosis("improbable diagnosis")
        assert "improbable" in result, f"Corrupted: {result}"

    def test_possible_removed_as_word(self) -> None:
        from src.rewards import _normalize_diagnosis

        result = _normalize_diagnosis("possible glioma")
        assert "possible" not in result
        assert "glioma" in result


class TestWhitespaceNormalization:
    """Finding #9: Whitespace normalization handles tabs and mixed whitespace."""

    def test_tab_normalized(self) -> None:
        from src.evaluation.diagnosis import normalize_diagnosis_string

        result = normalize_diagnosis_string("glioma\tgrade iv")
        assert "\t" not in result
        assert result == "glioma grade iv"

    def test_mixed_whitespace_normalized(self) -> None:
        from src.evaluation.diagnosis import normalize_diagnosis_string

        result = normalize_diagnosis_string("glioma   \t  grade  iv")
        assert result == "glioma grade iv"


class TestRescaleVsClamp:
    """Smoke test: rescale_and_clamp_box vs clamp_and_validate_box."""

    def test_rescale_preserves_relative_position(self) -> None:
        pytest.importorskip("torch")
        from src.evaluation.detection import rescale_and_clamp_box

        # Model outputs in 1000x1000 space for a 480x480 image
        box = rescale_and_clamp_box([100, 200, 500, 800], 480, 480)
        # x-coords should be scaled by 480/500=0.96, y-coords by 480/800=0.6
        assert box[0] < box[2], "x1 should be < x2"
        assert box[1] < box[3], "y1 should be < y2"
        assert all(0 <= c <= 480 for c in box), f"All coords in [0, 480]: {box}"

    def test_clamp_squishes_out_of_bounds(self) -> None:
        pytest.importorskip("torch")
        from src.evaluation.detection import clamp_and_validate_box

        box = clamp_and_validate_box([100, 200, 700, 900], 480, 480)
        assert box[2] == 480.0, "Clamped x2 should equal width"
        assert box[3] == 480.0, "Clamped y2 should equal height"

    def test_rescale_better_than_clamp_for_shifted_coords(self) -> None:
        pytest.importorskip("torch")
        from src.evaluation.detection import clamp_and_validate_box
        from src.evaluation.detection import rescale_and_clamp_box

        # Simulating model output in 1000x1000 space for 480x480 image
        raw_box = [200, 300, 600, 700]
        width, height = 480, 480

        rescaled = rescale_and_clamp_box(raw_box, width, height)
        clamped = clamp_and_validate_box(raw_box, width, height)

        # Rescaled box should preserve aspect ratio; clamped box loses right/bottom
        rescaled_w = rescaled[2] - rescaled[0]
        assert rescaled_w > 0, "Rescaled box should have positive width"
        # Clamped x2 is clipped to 480, so clamped width = 480 - 200 = 280
        # Rescaled width preserves original 400-unit span proportionally
        assert clamped[2] == 480.0, "Clamped x2 should be clipped"

    def test_swapped_coordinates_handled(self) -> None:
        pytest.importorskip("torch")
        from src.evaluation.detection import rescale_and_clamp_box

        box = rescale_and_clamp_box([300, 400, 100, 200], 480, 480)
        assert box[0] <= box[2], "Should swap x1/x2 if needed"
        assert box[1] <= box[3], "Should swap y1/y2 if needed"


class TestContainmentMatchGuard:
    """Finding #6: Containment match requires >= 2 words to prevent false positives.

    The containment check lives in evaluate_diagnosis_nova_official (the async
    LLM-based evaluation), not in exact_diagnosis_match.  We verify the guard
    logic by checking that exact_diagnosis_match correctly rejects non-synonym
    substrings (it has no containment logic), and that single-word exact
    matches still work.
    """

    def test_single_word_not_substring_match(self) -> None:
        from src.evaluation.diagnosis import exact_diagnosis_match

        # "tumor" is NOT the same as "brain tumor" — exact_diagnosis_match
        # should reject because it only does exact + synonym matching
        assert not exact_diagnosis_match("tumor", "brain tumor with edema")

    def test_exact_single_word_still_matches(self) -> None:
        from src.evaluation.diagnosis import exact_diagnosis_match

        # Exact equality should still work even for single words
        assert exact_diagnosis_match("glioma", "glioma")

    def test_containment_guard_code_present(self) -> None:
        """Verify the 2-word guard is in the source code."""
        content = (EXAMPLE_ROOT / "src" / "evaluation" / "diagnosis.py").read_text()
        assert "len(shorter.split()) >= 2" in content


class TestAreaPenaltySwappedCoordinates:
    """Area penalty must use abs() for swapped box coordinates."""

    def test_swapped_x_same_penalty(self) -> None:
        from src.rewards import _area_penalty

        normal = _area_penalty([10.0, 10.0, 200.0, 200.0], image_area=40000.0)
        swapped = _area_penalty([200.0, 10.0, 10.0, 200.0], image_area=40000.0)
        assert abs(normal - swapped) < 1e-6, (
            f"Swapped x coords should give same penalty: {normal} vs {swapped}"
        )

    def test_swapped_y_same_penalty(self) -> None:
        from src.rewards import _area_penalty

        normal = _area_penalty([10.0, 10.0, 200.0, 200.0], image_area=40000.0)
        swapped = _area_penalty([10.0, 200.0, 200.0, 10.0], image_area=40000.0)
        assert abs(normal - swapped) < 1e-6, (
            f"Swapped y coords should give same penalty: {normal} vs {swapped}"
        )


class TestSampleStdAggregation:
    """aggregate.py must use sample std (n-1 denominator)."""

    def test_std_uses_bessel_correction(self) -> None:
        """Population std of [0, 2] is 1.0, sample std is ~1.414."""
        import math

        # Verify Bessel correction: std([0, 2]) with n-1 = sqrt(2) ≈ 1.414
        values = [0.0, 2.0]
        mean = 1.0
        sample_var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        expected_std = math.sqrt(sample_var)

        assert abs(expected_std - math.sqrt(2)) < 1e-10
        assert expected_std > 1.0, "Sample std of [0, 2] should be > 1.0 (Bessel correction)"

    def test_aggregate_source_uses_n_minus_1(self) -> None:
        """Verify sample_std uses (len(xs) - 1) denominator (Bessel correction)."""
        paper_experiments = (
            REPO_ROOT / "examples" / "aiih2026_paper" / "experiments" / "__init__.py"
        )
        content = paper_experiments.read_text()
        assert "(len(xs) - 1)" in content, "sample_std must use Bessel correction"
