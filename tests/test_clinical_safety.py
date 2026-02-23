"""Clinical safety tests for patient-safety-critical code paths.

Tests cover:
1. Diagnosis matching — no clinically wrong equivalences
2. Threshold tool — minimum window width prevents destructive windowing
3. Schema validation — nested structure validation catches malformed outputs
4. IoU area penalty — reward hacking prevention via coordinate range evasion
5. Config bounds — safe defaults for medical imaging
"""

from __future__ import annotations

import pytest
from PIL import Image

from radiant_harness.config import ImageProcessingConfig
from radiant_harness.tools.visual import apply_intensity_threshold

# Guard imports that chain through evaluation/__init__.py → detection.py → torch
try:
    from examples.nova.src.evaluation.diagnosis import (
        exact_diagnosis_match,
        normalize_diagnosis_string,
    )

    _HAS_DIAGNOSIS = True
except (ImportError, ModuleNotFoundError):
    _HAS_DIAGNOSIS = False

from examples.nova.src.schemas import validate_nova_response

_skip_no_torch = pytest.mark.skipif(not _HAS_DIAGNOSIS, reason="torch not installed")


# =====================================================================
# 1. Diagnosis matching — clinically distinct conditions must NOT match
# =====================================================================


@_skip_no_torch
class TestDiagnosisMatchSafety:
    """Verify that clinically distinct diagnoses are NOT conflated."""

    def test_generic_cyst_not_equal_arachnoid_cyst(self) -> None:
        """Generic 'cyst' must not match 'arachnoid cyst' — they are different entities."""
        assert exact_diagnosis_match("cyst", "arachnoid cyst") is False

    def test_arachnoid_cyst_not_equal_generic_cyst(self) -> None:
        assert exact_diagnosis_match("arachnoid cyst", "cyst") is False

    def test_encephalitis_not_equal_meningitis(self) -> None:
        """Encephalitis and meningitis are distinct infections with different treatments."""
        assert exact_diagnosis_match("encephalitis", "meningitis") is False

    def test_contusion_not_equal_brain_injury(self) -> None:
        """Contusion is a specific type; 'brain injury' is a broader category."""
        assert exact_diagnosis_match("contusion", "brain injury") is False

    def test_communicating_hydrocephalus_not_equal_obstructive(self) -> None:
        """Different hydrocephalus types require different interventions."""
        assert (
            exact_diagnosis_match("communicating hydrocephalus", "obstructive hydrocephalus")
            is False
        )

    def test_medulloblastoma_not_equal_pnet(self) -> None:
        """Medulloblastoma and PNET are now classified as distinct entities (WHO 2021)."""
        assert exact_diagnosis_match("medulloblastoma", "pnet") is False

    def test_stroke_not_equal_cerebral_infarction(self) -> None:
        """'stroke' is broader (includes hemorrhagic); cerebral infarction is ischemic only.
        These should NOT match via exact_diagnosis_match (defer to LLM)."""
        assert exact_diagnosis_match("stroke", "cerebral infarction") is False

    # --- True synonyms SHOULD still match ---

    def test_acoustic_neuroma_equals_vestibular_schwannoma(self) -> None:
        """True synonym pair — same entity, different names."""
        assert exact_diagnosis_match("acoustic neuroma", "vestibular schwannoma") is True

    def test_cavernoma_equals_cavernous_malformation(self) -> None:
        assert exact_diagnosis_match("cavernoma", "cavernous malformation") is True

    def test_gbm_abbreviation(self) -> None:
        assert exact_diagnosis_match("glioblastoma", "glioblastoma multiforme") is True

    def test_acc_abbreviation(self) -> None:
        assert exact_diagnosis_match("agenesis of corpus callosum", "acc") is True

    def test_sah_abbreviation(self) -> None:
        assert exact_diagnosis_match("subarachnoid hemorrhage", "sah") is True

    def test_sod_abbreviation(self) -> None:
        assert exact_diagnosis_match("septo-optic dysplasia", "sod") is True

    def test_shearing_injury_equals_dai(self) -> None:
        assert exact_diagnosis_match("shearing injury", "diffuse axonal injury") is True


@_skip_no_torch
class TestDiagnosisNormalization:
    """Test normalization correctness."""

    def test_abbreviation_expansion(self) -> None:
        assert normalize_diagnosis_string("sod") == "septo-optic dysplasia"
        assert normalize_diagnosis_string("acc") == "agenesis of corpus callosum"

    def test_en_dash_normalization(self) -> None:
        assert normalize_diagnosis_string("septo\u2013optic") == "septo-optic"

    def test_empty_string(self) -> None:
        assert normalize_diagnosis_string("") == ""


# =====================================================================
# 2. Threshold tool — minimum window width
# =====================================================================


class TestThresholdWindowSafety:
    """Verify that narrow threshold windows are rejected."""

    def _make_gray_image(self) -> Image.Image:
        return Image.new("L", (100, 100), color=128)

    def test_narrow_window_rejected(self) -> None:
        """Window width of 5 (250-255) destroys diagnostic info — must raise."""
        img = self._make_gray_image()
        with pytest.raises(ValueError, match="below minimum"):
            apply_intensity_threshold(img, 250, 255)

    def test_narrow_window_rejected_custom_config(self) -> None:
        """Custom min_threshold_window is respected."""
        img = self._make_gray_image()
        cfg = ImageProcessingConfig(min_threshold_window=50)
        with pytest.raises(ValueError, match="below minimum"):
            apply_intensity_threshold(img, 100, 140, config=cfg)

    def test_valid_window_accepted(self) -> None:
        """Window of 100 (50-150) is wide enough — should succeed."""
        img = self._make_gray_image()
        result = apply_intensity_threshold(img, 50, 150)
        assert result.size == (100, 100)

    def test_full_range_accepted(self) -> None:
        """Full range 0-255 should always succeed."""
        img = self._make_gray_image()
        result = apply_intensity_threshold(img, 0, 255)
        assert result.size == (100, 100)

    def test_minimum_valid_window(self) -> None:
        """Window exactly at minimum (default=50) should succeed."""
        img = self._make_gray_image()
        result = apply_intensity_threshold(img, 100, 150)
        assert result.size == (100, 100)

    def test_one_below_minimum_fails(self) -> None:
        """Window one below minimum should fail."""
        img = self._make_gray_image()
        with pytest.raises(ValueError, match="below minimum"):
            apply_intensity_threshold(img, 100, 149)

    def test_config_validation(self) -> None:
        """min_threshold_window must be in [1, 255]."""
        with pytest.raises(ValueError, match="min_threshold_window"):
            ImageProcessingConfig(min_threshold_window=0)
        with pytest.raises(ValueError, match="min_threshold_window"):
            ImageProcessingConfig(min_threshold_window=256)


# =====================================================================
# 3. Schema validation — nested structure
# =====================================================================


def _make_valid_response() -> dict:
    """Minimal valid NOVA response."""
    return {
        "caption": {
            "description": "T2-weighted axial MRI showing hyperintense lesion",
            "sequence_characteristics": "T2W",
            "orientation": "axial",
            "confidence": 0.85,
        },
        "diagnosis": {
            "primary_diagnosis": "glioblastoma",
            "confidence": 0.7,
            "evidence": ["ring enhancement", "necrotic center"],
        },
        "localization": {
            "localizations": [
                {
                    "finding": "mass",
                    "bounding_box": [10, 20, 50, 60],
                    "anatomical_location": "right temporal lobe",
                    "confidence": 0.8,
                }
            ],
            "image_dimensions": {"width": 256, "height": 256},
            "coordinate_system": "absolute_pixels",
        },
        "continue": False,
        "reasoning": "Test chain-of-thought reasoning",
    }


class TestSchemaValidationSafety:
    """Verify that malformed responses are caught."""

    def test_valid_response_passes(self) -> None:
        assert validate_nova_response(_make_valid_response()) is True

    def test_missing_top_level_field(self) -> None:
        resp = _make_valid_response()
        del resp["diagnosis"]
        assert validate_nova_response(resp) is False

    def test_caption_is_null(self) -> None:
        """caption: null should fail — not a dict."""
        resp = _make_valid_response()
        resp["caption"] = None
        assert validate_nova_response(resp) is False

    def test_diagnosis_is_integer(self) -> None:
        """diagnosis: 42 should fail — not a dict."""
        resp = _make_valid_response()
        resp["diagnosis"] = 42
        assert validate_nova_response(resp) is False

    def test_localization_is_string(self) -> None:
        resp = _make_valid_response()
        resp["localization"] = ""
        assert validate_nova_response(resp) is False

    def test_caption_missing_description(self) -> None:
        """caption without 'description' string should fail."""
        resp = _make_valid_response()
        resp["caption"] = {"sequence_characteristics": "T2W"}
        assert validate_nova_response(resp) is False

    def test_diagnosis_missing_primary(self) -> None:
        """diagnosis without 'primary_diagnosis' string should fail."""
        resp = _make_valid_response()
        resp["diagnosis"] = {"confidence": 0.5, "evidence": []}
        assert validate_nova_response(resp) is False

    def test_localization_missing_localizations_list(self) -> None:
        resp = _make_valid_response()
        resp["localization"] = {"image_dimensions": {"width": 256, "height": 256}}
        assert validate_nova_response(resp) is False

    def test_continue_not_bool(self) -> None:
        resp = _make_valid_response()
        resp["continue"] = "false"
        assert validate_nova_response(resp) is False

    def test_continue_true_is_valid(self) -> None:
        resp = _make_valid_response()
        resp["continue"] = True
        assert validate_nova_response(resp) is True


# =====================================================================
# 4. IoU area penalty — reward hacking prevention
# =====================================================================


class TestIoUAreaPenaltyBypass:
    """Verify the area penalty cannot be evaded by coordinate range mismatch.

    Before fix: IoUReward(normalized=True) with pixel-scale coords set
    image_area=0 and silently skipped the penalty.  A model could game
    reward by predicting full-image pixel boxes.
    """

    def test_normalized_mode_pixel_coords_still_penalized(self) -> None:
        """When normalized=True but model outputs pixel coords,
        the area penalty must still apply."""
        from radiant_harness.verifiers.rewards import IoUReward

        reward_fn = IoUReward(normalized=True, continuous=True, area_penalty_start=0.5)

        info = {"bbox": [0, 0, 512, 512]}
        completion = "[0, 0, 512, 512]"
        reward = reward_fn("prompt", completion, info)

        # IoU=1.0, area_ratio≈1.0 → penalty should drive reward near 0
        assert reward < 0.5, (
            f"Full-image pixel box got reward={reward} in normalized mode. "
            f"Area penalty was bypassed."
        )

    def test_normalized_mode_valid_coords_penalized(self) -> None:
        """Full-image box in [0,1] range gets full penalty."""
        from radiant_harness.verifiers.rewards import IoUReward

        reward_fn = IoUReward(normalized=True, continuous=True, area_penalty_start=0.5)

        info = {"bbox": [0.0, 0.0, 1.0, 1.0]}
        completion = "[0.0, 0.0, 1.0, 1.0]"
        reward = reward_fn("prompt", completion, info)

        assert reward == 0.0, (
            f"Full-image normalized box should get reward=0.0, got {reward}"
        )

    def test_normalized_mode_small_box_no_penalty(self) -> None:
        """Small boxes below penalty_start should not be penalized."""
        from radiant_harness.verifiers.rewards import IoUReward

        reward_fn = IoUReward(normalized=True, continuous=True, area_penalty_start=0.5)

        info = {"bbox": [0.2, 0.2, 0.3, 0.3]}
        completion = "[0.2, 0.2, 0.3, 0.3]"
        reward = reward_fn("prompt", completion, info)

        assert reward == 1.0, (
            f"Small box with perfect IoU should get reward=1.0, got {reward}"
        )


# =====================================================================
# 5. Config bounds — safe defaults for medical imaging
# =====================================================================


class TestConfigBoundsDefaults:
    """Verify default config bounds prevent diagnostic information destruction."""

    def test_min_threshold_window_default_at_least_50(self) -> None:
        """A 30-unit window (11.8% of 8-bit range) destroys subtle lesion
        contrast. Default must be >= 50 for brain MRI safety."""
        from radiant_harness.config import get_config

        cfg = get_config()
        assert cfg.image.min_threshold_window >= 50

    def test_min_window_width_default_at_least_10(self) -> None:
        """A 2-unit window reduces 8-bit images to near-binary."""
        from radiant_harness.config import get_config

        cfg = get_config()
        assert cfg.image.min_window_width >= 10


# =====================================================================
# 6. Schema validation — element-level localization checks (Patch Set 2)
# =====================================================================


class TestSchemaValidationLocalizationElements:
    """Verify that validate_nova_response catches malformed localization elements.

    Before PS2: localizations could be any list — [{"garbage": 1}] passed.
    After PS2: each element must have finding (str), bounding_box (4 nums),
    anatomical_location (str), confidence (0-1 float).
    """

    def test_empty_localizations_list_passes(self) -> None:
        """Empty list is valid — model found no abnormalities."""
        resp = _make_valid_response()
        resp["localization"]["localizations"] = []
        assert validate_nova_response(resp) is True

    def test_garbage_element_rejected(self) -> None:
        """An element without required keys must fail."""
        resp = _make_valid_response()
        resp["localization"]["localizations"] = [{"garbage": True}]
        assert validate_nova_response(resp) is False

    def test_missing_finding_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"] = [
            {
                "bounding_box": [10, 20, 50, 60],
                "anatomical_location": "frontal lobe",
                "confidence": 0.8,
            }
        ]
        assert validate_nova_response(resp) is False

    def test_missing_bounding_box_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"] = [
            {
                "finding": "mass",
                "anatomical_location": "frontal lobe",
                "confidence": 0.8,
            }
        ]
        assert validate_nova_response(resp) is False

    def test_missing_anatomical_location_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"] = [
            {
                "finding": "mass",
                "bounding_box": [10, 20, 50, 60],
                "confidence": 0.8,
            }
        ]
        assert validate_nova_response(resp) is False

    def test_missing_confidence_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"] = [
            {
                "finding": "mass",
                "bounding_box": [10, 20, 50, 60],
                "anatomical_location": "frontal lobe",
            }
        ]
        assert validate_nova_response(resp) is False


class TestSchemaValidationBoundingBox:
    """Verify bounding_box must be exactly 4 finite numbers."""

    def test_three_element_bbox_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["bounding_box"] = [10, 20, 50]
        assert validate_nova_response(resp) is False

    def test_five_element_bbox_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["bounding_box"] = [10, 20, 50, 60, 99]
        assert validate_nova_response(resp) is False

    def test_one_element_bbox_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["bounding_box"] = [10]
        assert validate_nova_response(resp) is False

    def test_empty_bbox_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["bounding_box"] = []
        assert validate_nova_response(resp) is False

    def test_string_in_bbox_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["bounding_box"] = [10, "twenty", 50, 60]
        assert validate_nova_response(resp) is False

    def test_nan_in_bbox_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["bounding_box"] = [10, float("nan"), 50, 60]
        assert validate_nova_response(resp) is False

    def test_inf_in_bbox_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["bounding_box"] = [10, float("inf"), 50, 60]
        assert validate_nova_response(resp) is False

    def test_bool_in_bbox_rejected(self) -> None:
        """bool is technically a subclass of int in Python — must be excluded."""
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["bounding_box"] = [True, False, True, False]
        assert validate_nova_response(resp) is False

    def test_valid_four_element_bbox_passes(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["bounding_box"] = [0, 0, 512, 512]
        assert validate_nova_response(resp) is True

    def test_float_bbox_passes(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["bounding_box"] = [10.5, 20.3, 50.7, 60.1]
        assert validate_nova_response(resp) is True


class TestSchemaValidationConfidence:
    """Verify confidence must be a finite number in [0.0, 1.0]."""

    def test_confidence_above_one_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["confidence"] = 1.5
        assert validate_nova_response(resp) is False

    def test_confidence_negative_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["confidence"] = -0.1
        assert validate_nova_response(resp) is False

    def test_confidence_ninety_nine_rejected(self) -> None:
        """Model outputting confidence: 99.0 (percentage) must be caught."""
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["confidence"] = 99.0
        assert validate_nova_response(resp) is False

    def test_confidence_nan_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["confidence"] = float("nan")
        assert validate_nova_response(resp) is False

    def test_confidence_inf_rejected(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["confidence"] = float("inf")
        assert validate_nova_response(resp) is False

    def test_confidence_bool_rejected(self) -> None:
        """bool is technically int in Python — confidence: True should fail."""
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["confidence"] = True
        assert validate_nova_response(resp) is False

    def test_confidence_zero_passes(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["confidence"] = 0.0
        assert validate_nova_response(resp) is True

    def test_confidence_one_passes(self) -> None:
        resp = _make_valid_response()
        resp["localization"]["localizations"][0]["confidence"] = 1.0
        assert validate_nova_response(resp) is True

    def test_caption_confidence_above_one_rejected(self) -> None:
        resp = _make_valid_response()
        resp["caption"]["confidence"] = 5.0
        assert validate_nova_response(resp) is False

    def test_diagnosis_confidence_above_one_rejected(self) -> None:
        resp = _make_valid_response()
        resp["diagnosis"]["confidence"] = 2.0
        assert validate_nova_response(resp) is False

    def test_caption_confidence_required(self) -> None:
        """confidence is required in caption — missing should fail."""
        resp = _make_valid_response()
        del resp["caption"]["confidence"]
        assert validate_nova_response(resp) is False


class TestSchemaValidationDifferentialDiagnoses:
    """Verify differential_diagnoses elements are validated."""

    def test_diff_missing_diagnosis_string_rejected(self) -> None:
        resp = _make_valid_response()
        resp["diagnosis"]["differential_diagnoses"] = [
            {"confidence": 0.5}
        ]
        assert validate_nova_response(resp) is False

    def test_diff_missing_confidence_rejected(self) -> None:
        resp = _make_valid_response()
        resp["diagnosis"]["differential_diagnoses"] = [
            {"diagnosis": "meningioma"}
        ]
        assert validate_nova_response(resp) is False

    def test_diff_confidence_above_one_rejected(self) -> None:
        resp = _make_valid_response()
        resp["diagnosis"]["differential_diagnoses"] = [
            {"diagnosis": "meningioma", "confidence": 1.5}
        ]
        assert validate_nova_response(resp) is False

    def test_diff_not_a_dict_rejected(self) -> None:
        resp = _make_valid_response()
        resp["diagnosis"]["differential_diagnoses"] = ["meningioma"]
        assert validate_nova_response(resp) is False

    def test_valid_differential_passes(self) -> None:
        resp = _make_valid_response()
        resp["diagnosis"]["differential_diagnoses"] = [
            {"diagnosis": "meningioma", "confidence": 0.3},
            {"diagnosis": "metastasis", "confidence": 0.2},
        ]
        assert validate_nova_response(resp) is True

    def test_no_differentials_passes(self) -> None:
        """differential_diagnoses is optional — absent should pass."""
        resp = _make_valid_response()
        # Not present in _make_valid_response by default
        assert "differential_diagnoses" not in resp["diagnosis"]
        assert validate_nova_response(resp) is True


class TestSchemaValidationEvidence:
    """Verify evidence list elements are validated."""

    def test_evidence_with_non_string_rejected(self) -> None:
        resp = _make_valid_response()
        resp["diagnosis"]["evidence"] = ["ring enhancement", 42]
        assert validate_nova_response(resp) is False

    def test_evidence_not_a_list_rejected(self) -> None:
        resp = _make_valid_response()
        resp["diagnosis"]["evidence"] = "ring enhancement"
        assert validate_nova_response(resp) is False

    def test_evidence_empty_list_passes(self) -> None:
        resp = _make_valid_response()
        resp["diagnosis"]["evidence"] = []
        assert validate_nova_response(resp) is True

    def test_evidence_absent_passes(self) -> None:
        """evidence is optional — absent should pass."""
        resp = _make_valid_response()
        del resp["diagnosis"]["evidence"]
        assert validate_nova_response(resp) is True
