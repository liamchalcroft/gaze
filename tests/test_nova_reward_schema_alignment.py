"""Regression tests for NOVA reward/schema alignment.

Tests both the examples/nova rewards and environments/nova_brain_mri rewards
to ensure parity between the two implementations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Import examples/nova/src as package "src"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_NOVA_ROOT = REPO_ROOT / "examples" / "nova"
if str(EXAMPLE_NOVA_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_NOVA_ROOT))

# Also import environments/nova_brain_mri/src for parity testing
ENV_NOVA_ROOT = REPO_ROOT / "environments" / "nova_brain_mri" / "src"
if str(ENV_NOVA_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_NOVA_ROOT))

from src.rewards import NOVAVerifiersReward
from src.rewards import compute_localization_reward


def _schema_completion() -> str:
    return json.dumps(
        {
            "caption": {
                "description": "left temporal lobe lesion",
                "sequence_characteristics": "T2",
                "orientation": "axial",
                "confidence": 0.9,
            },
            "diagnosis": {
                "primary_diagnosis": "glioma",
                "confidence": 0.8,
                "evidence": ["left temporal lesion"],
                "differential_diagnoses": [],
            },
            "localization": {
                "localizations": [
                    {
                        "finding": "lesion",
                        "bounding_box": [10, 10, 20, 20],
                        "anatomical_location": "left temporal lobe",
                        "confidence": 0.8,
                    }
                ],
                "image_dimensions": {"width": 64, "height": 64},
                "coordinate_system": "absolute_pixels",
            },
            "continue": False,
        }
    )


def test_nova_rewards_use_schema_keys() -> None:
    completion = _schema_completion()
    info = {
        "caption": "left temporal lobe lesion",
        "diagnosis": "glioma",
        "boxes": [[10, 10, 20, 20]],
    }

    assert NOVAVerifiersReward(task="caption")("", completion, info) == 1.0
    assert NOVAVerifiersReward(task="diagnosis")("", completion, info) == 1.0
    assert NOVAVerifiersReward(task="localization")("", completion, info) == 1.0
    assert NOVAVerifiersReward(task="all")("", completion, info) == 1.0


def test_localization_reward_penalizes_false_positive_without_reference() -> None:
    # No lesions in reference: spurious predictions should not get credit.
    assert compute_localization_reward([[0, 0, 10, 10]], []) == 0.0
    # Empty prediction with empty reference should still be treated as correct.
    assert compute_localization_reward([], []) == 1.0


# ---------------------------------------------------------------------------
# Parity tests: environments/nova_brain_mri vs examples/nova
# ---------------------------------------------------------------------------


class TestRewardParity:
    """Verify environments/nova_brain_mri rewards produce identical results
    to examples/nova rewards on the same inputs.
    """

    @pytest.fixture
    def env_rewards(self):
        """Import the environment rewards module."""
        from nova_brain_mri import rewards as env_rw

        return env_rw

    @pytest.fixture
    def example_rewards(self):
        """Import the example rewards module."""
        from src import rewards as ex_rw

        return ex_rw

    @pytest.fixture
    def sample_completion(self) -> str:
        return _schema_completion()

    @pytest.fixture
    def sample_info(self) -> dict:
        return {
            "caption": "left temporal lobe lesion",
            "diagnosis": "glioma",
            "boxes": [[10, 10, 20, 20]],
        }

    def test_caption_parity(self, env_rewards, sample_completion, sample_info) -> None:
        env_score = env_rewards.caption_reward("", sample_completion, sample_info)
        # Compute same thing via examples path
        from src.rewards import NOVAVerifiersReward

        ex_score = NOVAVerifiersReward(task="caption")("", sample_completion, sample_info)
        assert env_score == pytest.approx(ex_score), (
            f"Caption parity failed: env={env_score}, example={ex_score}"
        )

    def test_diagnosis_parity(self, env_rewards, sample_completion, sample_info) -> None:
        env_score = env_rewards.diagnosis_reward("", sample_completion, sample_info)
        from src.rewards import NOVAVerifiersReward

        ex_score = NOVAVerifiersReward(task="diagnosis")("", sample_completion, sample_info)
        assert env_score == pytest.approx(ex_score), (
            f"Diagnosis parity failed: env={env_score}, example={ex_score}"
        )

    def test_localization_parity(self, env_rewards, sample_completion, sample_info) -> None:
        loc_fn = env_rewards.localization_reward_factory(iou_threshold=0.5)
        env_score = loc_fn("", sample_completion, sample_info)
        from src.rewards import NOVAVerifiersReward

        ex_score = NOVAVerifiersReward(task="localization")("", sample_completion, sample_info)
        assert env_score == pytest.approx(ex_score), (
            f"Localization parity failed: env={env_score}, example={ex_score}"
        )

    def test_severity_qualifiers_preserved(self, env_rewards) -> None:
        """Severity qualifiers (mild, moderate, severe) must NOT be stripped.

        These are clinically meaningful and change the diagnosis.
        Stripping them would equate 'mild hydrocephalus' with 'severe
        hydrocephalus', which is a critical clinical error.
        """
        completion_mild = json.dumps({"diagnosis": {"primary_diagnosis": "mild hydrocephalus"}})
        info_mild = {"diagnosis": "mild hydrocephalus"}
        info_severe = {"diagnosis": "severe hydrocephalus"}

        # mild vs mild → should match
        score_correct = env_rewards.diagnosis_reward("", completion_mild, info_mild)
        assert score_correct == 1.0, f"Expected 1.0, got {score_correct}"

        # mild vs severe → should NOT match (severity matters)
        score_wrong = env_rewards.diagnosis_reward("", completion_mild, info_severe)
        assert score_wrong < 1.0, (
            f"Expected <1.0 for mild vs severe, got {score_wrong}. "
            "Severity qualifiers are being incorrectly stripped."
        )

    def test_iou_threshold_defaults_to_0_5(self, env_rewards) -> None:
        """Default IoU threshold must be 0.5 to match NOVA eval (ACC50)."""
        # Box pair with IoU ~0.35 — should fail at 0.5 threshold
        completion = json.dumps(
            {
                "localization": [{"bounding_box": [0, 0, 10, 10]}],
            }
        )
        info = {"boxes": [[5, 5, 15, 15]]}

        loc_fn = env_rewards.localization_reward_factory()  # default threshold
        score = loc_fn("", completion, info)
        # IoU of [0,0,10,10] vs [5,5,15,15] = 25/175 ≈ 0.143 < 0.5
        assert score == 0.0, (
            f"Expected 0.0 for IoU≈0.14 at default threshold, got {score}. "
            "Default iou_threshold may not be 0.5."
        )

    def test_markdown_json_extraction(self, env_rewards) -> None:
        """JSON wrapped in markdown code blocks must be parsed correctly."""
        completion = '```json\n{"caption": "test finding"}\n```'
        info = {"caption": "test finding"}
        score = env_rewards.caption_reward("", completion, info)
        assert score == 1.0, (
            f"Expected 1.0 for markdown-wrapped JSON, got {score}. "
            "JSON extraction may not handle markdown code blocks."
        )

    def test_iou_coordinate_normalization(self) -> None:
        """Core compute_iou normalizes swapped coordinates; env must too."""
        from radiant_harness.utils.iou import compute_iou

        # Swapped coords: x1 > x2
        box_normal = [0.0, 0.0, 10.0, 10.0]
        box_swapped = [10.0, 10.0, 0.0, 0.0]
        assert compute_iou(box_normal, box_swapped) == 1.0
