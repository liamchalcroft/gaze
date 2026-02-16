"""Clinical safety tests for patient-safety-critical code paths.

Tests cover:
1. Diagnosis matching — no clinically wrong equivalences
2. Threshold tool — minimum window width prevents destructive windowing
3. Schema validation — nested structure validation catches malformed outputs
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
        """Window exactly at minimum (default=30) should succeed."""
        img = self._make_gray_image()
        result = apply_intensity_threshold(img, 100, 130)
        assert result.size == (100, 100)

    def test_one_below_minimum_fails(self) -> None:
        """Window one below minimum should fail."""
        img = self._make_gray_image()
        with pytest.raises(ValueError, match="below minimum"):
            apply_intensity_threshold(img, 100, 129)

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
