"""Tests for Patch Set 2: Code correctness fixes.

Covers:
1. IoUReward._extract_bbox searches all JSON objects (not just first)
2. _maybe_normalize_box handles zero coordinates at image edge
3. Diagnosis reward expands abbreviations (aligned with evaluation)
"""

from __future__ import annotations

import json

from PIL import Image

from examples.nova.src.rewards import _normalize_diagnosis
from examples.nova.src.rewards import compute_diagnosis_reward
from radiant_harness.tools.visual import _maybe_normalize_box
from radiant_harness.tools.visual import _maybe_normalize_point
from radiant_harness.verifiers.rewards import IoUReward

# =====================================================================
# 1. IoUReward._extract_bbox — search all JSON objects
# =====================================================================


class TestExtractBboxAllJsonObjects:
    """Verify _extract_bbox searches beyond the first JSON object."""

    def _make_info(self, bbox: list[float]) -> dict:
        return {"bbox": bbox}

    def test_bbox_in_first_json(self) -> None:
        """bbox in first JSON object should still work."""
        reward = IoUReward(continuous=True, area_penalty_start=1.0)
        completion = json.dumps({"bbox": [10, 20, 30, 40]})
        score = reward("", completion, self._make_info([10, 20, 30, 40]))
        assert score == 1.0

    def test_bbox_in_second_json(self) -> None:
        """bbox in second JSON object should now be found.

        Previously, the `break` after the first JSON object meant this
        would return [] and score 0.0.
        """
        reward = IoUReward(continuous=True, area_penalty_start=1.0)
        # First JSON has no bbox, second has it
        completion = (
            json.dumps({"reasoning": "some analysis"})
            + "\n"
            + json.dumps({"bbox": [10, 20, 30, 40]})
        )
        score = reward("", completion, self._make_info([10, 20, 30, 40]))
        assert score == 1.0

    def test_bbox_in_nested_location(self) -> None:
        """bbox nested under location key should be found."""
        reward = IoUReward(continuous=True, area_penalty_start=1.0)
        completion = json.dumps({"location": {"bbox": [10, 20, 30, 40]}})
        score = reward("", completion, self._make_info([10, 20, 30, 40]))
        assert score == 1.0

    def test_bbox_in_localization_structure(self) -> None:
        """bbox in NOVA-style localization.localizations[0].bounding_box."""
        reward = IoUReward(continuous=True, area_penalty_start=1.0)
        completion = json.dumps(
            {
                "localization": {
                    "localizations": [
                        {
                            "finding": "lesion",
                            "bounding_box": [10, 20, 30, 40],
                        }
                    ]
                }
            }
        )
        score = reward("", completion, self._make_info([10, 20, 30, 40]))
        assert score == 1.0

    def test_no_bbox_anywhere(self) -> None:
        """Multiple JSON objects without bbox should return 0.0."""
        reward = IoUReward(continuous=True, area_penalty_start=1.0)
        completion = (
            json.dumps({"reasoning": "analysis"}) + "\n" + json.dumps({"diagnosis": "glioma"})
        )
        score = reward("", completion, self._make_info([10, 20, 30, 40]))
        assert score == 0.0

    def test_first_json_invalid_second_has_bbox(self) -> None:
        """Invalid first JSON followed by valid second should find bbox."""
        reward = IoUReward(continuous=True, area_penalty_start=1.0)
        # Malformed first JSON, valid second
        completion = "{invalid json}" + "\n" + json.dumps({"bbox": [10, 20, 30, 40]})
        score = reward("", completion, self._make_info([10, 20, 30, 40]))
        assert score == 1.0

    def test_regex_fallback_still_works(self) -> None:
        """When no JSON objects found, regex fallback for [x1,y1,x2,y2] works."""
        reward = IoUReward(continuous=True, area_penalty_start=1.0)
        completion = "The bounding box is [10, 20, 30, 40] for the lesion."
        score = reward("", completion, self._make_info([10, 20, 30, 40]))
        assert score == 1.0


# =====================================================================
# 2. _maybe_normalize_box — handle zero coordinates
# =====================================================================


class TestMaybeNormalizeBoxZeroCoords:
    """Verify _maybe_normalize_box handles boxes starting at image edge."""

    def _make_image(self, width: int = 480, height: int = 480) -> Image.Image:
        return Image.new("L", (width, height))

    def test_pixel_box_with_zero_x1(self) -> None:
        """[0, 50, 200, 300] is pixel coords — should be normalized.

        Previously failed because all(v > 1 ...) is False when x1=0.
        """
        img = self._make_image(480, 480)
        result = _maybe_normalize_box([0.0, 50.0, 200.0, 300.0], img)
        assert all(0.0 <= v <= 1.0 for v in result), f"Expected normalized, got {result}"
        assert abs(result[0] - 0.0) < 1e-6  # x1 = 0/480
        assert abs(result[1] - 50.0 / 480) < 1e-6
        assert abs(result[2] - 200.0 / 480) < 1e-6
        assert abs(result[3] - 300.0 / 480) < 1e-6

    def test_pixel_box_with_zero_y1(self) -> None:
        """[50, 0, 200, 300] — y1=0, should still normalize."""
        img = self._make_image(480, 480)
        result = _maybe_normalize_box([50.0, 0.0, 200.0, 300.0], img)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_pixel_box_with_zero_x1_y1(self) -> None:
        """[0, 0, 200, 300] — top-left corner, should normalize."""
        img = self._make_image(480, 480)
        result = _maybe_normalize_box([0.0, 0.0, 200.0, 300.0], img)
        assert all(0.0 <= v <= 1.0 for v in result)
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_already_normalized_box_unchanged(self) -> None:
        """[0.2, 0.3, 0.8, 0.9] — already in [0,1], should pass through."""
        img = self._make_image(480, 480)
        box = [0.2, 0.3, 0.8, 0.9]
        result = _maybe_normalize_box(box, img)
        assert result == box

    def test_all_pixel_coords_above_one(self) -> None:
        """[50, 50, 200, 300] — all > 1, should normalize (pre-existing behavior)."""
        img = self._make_image(480, 480)
        result = _maybe_normalize_box([50.0, 50.0, 200.0, 300.0], img)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_point_with_zero_x(self) -> None:
        """_maybe_normalize_point should also handle zero coords."""
        img = self._make_image(480, 480)
        result = _maybe_normalize_point([0.0, 240.0], img)
        assert all(0.0 <= v <= 1.0 for v in result)
        assert result[0] == 0.0
        assert abs(result[1] - 0.5) < 1e-6

    def test_normalized_point_unchanged(self) -> None:
        """[0.5, 0.5] — already normalized, pass through."""
        img = self._make_image(480, 480)
        result = _maybe_normalize_point([0.5, 0.5], img)
        assert result == [0.5, 0.5]


# =====================================================================
# 3. Diagnosis reward — abbreviation expansion
# =====================================================================


class TestDiagnosisNormalizationAlignment:
    """Verify _normalize_diagnosis now expands medical abbreviations."""

    def test_gbm_expansion(self) -> None:
        """'gbm' should expand to 'glioblastoma multiforme'."""
        assert _normalize_diagnosis("GBM") == "glioblastoma multiforme"

    def test_sod_expansion(self) -> None:
        """'SOD' should expand to 'septo-optic dysplasia'."""
        assert _normalize_diagnosis("SOD") == "septo-optic dysplasia"

    def test_avm_expansion(self) -> None:
        """'AVM' should expand to 'arteriovenous malformation'."""
        assert _normalize_diagnosis("AVM") == "arteriovenous malformation"

    def test_sah_expansion(self) -> None:
        """'SAH' should expand to 'subarachnoid hemorrhage'."""
        assert _normalize_diagnosis("SAH") == "subarachnoid hemorrhage"

    def test_abbrev_with_suffix(self) -> None:
        """'GBM with necrosis' should expand the prefix."""
        result = _normalize_diagnosis("GBM with necrosis")
        assert result.startswith("glioblastoma multiforme")
        assert "necrosis" in result

    def test_en_dash_normalized(self) -> None:
        """En-dash should be converted to hyphen."""
        assert "–" not in _normalize_diagnosis("septo–optic dysplasia")
        assert "-" in _normalize_diagnosis("septo–optic dysplasia")

    def test_hedging_still_stripped(self) -> None:
        """Hedging modifiers should still be removed."""
        result = _normalize_diagnosis("possible glioblastoma")
        assert "possible" not in result
        assert "glioblastoma" in result

    def test_severity_preserved(self) -> None:
        """Severity qualifiers should NOT be stripped."""
        result = _normalize_diagnosis("severe hydrocephalus")
        assert "severe" in result


class TestDiagnosisRewardAbbreviations:
    """Verify compute_diagnosis_reward matches abbreviated vs full form."""

    def test_gbm_matches_full_form(self) -> None:
        """'GBM' should match 'glioblastoma multiforme'."""
        score = compute_diagnosis_reward("GBM", "glioblastoma multiforme")
        assert score > 0.5, f"GBM should match glioblastoma multiforme, got {score}"

    def test_sah_matches_full_form(self) -> None:
        """'SAH' should match 'subarachnoid hemorrhage'."""
        score = compute_diagnosis_reward("SAH", "subarachnoid hemorrhage")
        assert score > 0.5

    def test_avm_matches_full_form(self) -> None:
        """'AVM' should match 'arteriovenous malformation'."""
        score = compute_diagnosis_reward("AVM", "arteriovenous malformation")
        assert score > 0.5

    def test_nonmatching_diagnoses(self) -> None:
        """Completely different diagnoses should score 0."""
        score = compute_diagnosis_reward("meningioma", "glioblastoma")
        assert score == 0.0
