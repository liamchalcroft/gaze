"""Tests for NOVA audit patches."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "nova"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


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
    def test_prompt_includes_findings_field(self) -> None:
        content = (EXAMPLE_ROOT / "src" / "prompts" / "single_turn" / "task.jinja").read_text()
        assert '"findings"' in content

    def test_prompt_includes_anatomical_regions_field(self) -> None:
        content = (EXAMPLE_ROOT / "src" / "prompts" / "single_turn" / "task.jinja").read_text()
        assert '"anatomical_regions"' in content

    def test_prompt_json_example_matches_schema(self) -> None:
        from src.schemas import NOVA_SCHEMA

        content = (EXAMPLE_ROOT / "src" / "prompts" / "single_turn" / "task.jinja").read_text()
        schema_root = NOVA_SCHEMA["json_schema"]["schema"]
        caption_props = schema_root["properties"]["caption"]["properties"]
        for field_name in caption_props:
            assert f'"{field_name}"' in content, f"task.jinja missing caption field: {field_name}"


class TestGTBoxClampingDimensions:
    def test_gt_dimensions_passed_through_info(self) -> None:
        from src.rewards import NOVAVerifiersReward

        reward_fn = NOVAVerifiersReward(task="localization")
        completion = (
            '{"localization": {"localizations": ['
            '{"finding": "test", "bounding_box": [10, 10, 50, 50], '
            '"anatomical_location": "test", "confidence": 0.9}]}}'
        )

        info_with_dims = {
            "localizations": [{"bbox": [10, 10, 50, 50]}],
            "image_width": 256,
            "image_height": 256,
        }
        assert reward_fn("prompt", completion, info_with_dims) == 1.0

        info_without_dims = {"localizations": [{"bbox": [10, 10, 50, 50]}]}
        assert reward_fn("prompt", completion, info_without_dims) == 1.0

    def test_area_penalty_matters_for_large_boxes(self) -> None:
        from src.rewards import NOVAVerifiersReward

        reward_fn = NOVAVerifiersReward(task="localization")
        completion = (
            '{"localization": {"localizations": ['
            '{"finding": "test", "bounding_box": [0, 0, 95, 95], '
            '"anatomical_location": "test", "confidence": 0.9}]}}'
        )

        info_with_dims = {
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

        import torch

        from src.evaluation.caption import evaluate_caption

        # Simulate a very poor BERTScore (negative after baseline rescaling)
        negative_f1 = torch.tensor([-0.1, -0.2, -0.15])
        fake_result = (torch.zeros(3), torch.zeros(3), negative_f1)

        with patch("src.evaluation.caption.bert_score_fn", return_value=fake_result):
            result = evaluate_caption(["bad", "bad", "bad"], ["good ref", "good ref", "good ref"])

        assert result["bert_f1"] >= 0.0, f"bert_f1 should be >= 0, got {result['bert_f1']}"
        assert result["bert_f1"] <= 1.0, f"bert_f1 should be <= 1, got {result['bert_f1']}"

    def test_bert_f1_above_one_clamped(self) -> None:
        """Edge case: if BERTScore somehow exceeds 1.0, clamp it."""
        from unittest.mock import patch

        import torch

        from src.evaluation.caption import evaluate_caption

        high_f1 = torch.tensor([1.5, 1.2, 1.3])
        fake_result = (torch.zeros(3), torch.zeros(3), high_f1)

        with patch("src.evaluation.caption.bert_score_fn", return_value=fake_result):
            result = evaluate_caption(["test", "test", "test"], ["ref", "ref", "ref"])

        assert result["bert_f1"] <= 1.0, f"bert_f1 should be <= 1, got {result['bert_f1']}"


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
        info = {
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
        info = {
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
        info = {
            "localizations": [{"bbox": [10, 10, 50, 50]}],
            "image_width": 256,
            "image_height": 256,
        }
        assert reward_fn("prompt", completion, info) == 1.0


class TestSchemaValidationConfidence:
    """Finding #5: Schema validation should reject None confidence."""

    def test_null_caption_confidence_rejected(self) -> None:
        from src.schemas import validate_nova_response

        response = {
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

        response = {
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


class TestRequiredFieldsIncludesReasoning:
    """Finding #8: get_required_fields() must include 'reasoning'."""

    def test_reasoning_in_required_fields(self) -> None:
        from src.schemas import NOVA_SCHEMA
        from src.schemas import get_required_fields

        required = get_required_fields()
        assert "reasoning" in required

        schema_required = set(NOVA_SCHEMA["json_schema"]["schema"]["required"])
        assert set(required) == schema_required, (
            f"Mismatch: get_required_fields={set(required)} vs schema={schema_required}"
        )

    def test_response_without_reasoning_rejected(self) -> None:
        from src.schemas import validate_nova_response

        response = {
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
            # "reasoning" intentionally missing
        }
        assert not validate_nova_response(response), "Missing reasoning should fail"


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
