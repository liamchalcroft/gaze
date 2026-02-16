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
