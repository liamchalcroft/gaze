"""Cross-implementation parity tests for reward functions.

Verifies that _normalize_diagnosis, compute_iou, and extract_completion_text
produce identical results across all implementations in the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helper: import the standalone environment module without requiring install
# ---------------------------------------------------------------------------

ENV_SRC = Path(__file__).resolve().parent.parent / "environments" / "nova_brain_mri" / "src"
EXAMPLE_ROOT = Path(__file__).resolve().parent.parent / "examples" / "nova"


# ===========================================================================
# extract_completion_text parity
# ===========================================================================


_MULTIMODAL_COMPLETION = [
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "reasoning step"},
            {"type": "text", "text": '{"answer": "glioma"}'},
        ],
    }
]

_STRING_COMPLETION = [
    {"role": "assistant", "content": "simple string answer"},
]

_EMPTY_COMPLETION = [
    {"role": "system", "content": "system message only"},
]


# ===========================================================================
# compute_iou parity
# ===========================================================================


class TestComputeIoUParity:
    """GEMeX's compute_iou clamps to image bounds, then defers to the shared one.

    On well-formed, in-bounds boxes the two must agree *exactly*; the previous
    0.05 tolerance was wide enough to hide the divergence it claimed to detect.
    """

    @pytest.mark.parametrize(
        "box_a,box_b",
        [
            ([0, 0, 100, 100], [0, 0, 100, 100]),  # perfect overlap
            ([0, 0, 10, 10], [200, 200, 300, 300]),  # no overlap
            ([0, 0, 50, 50], [25, 25, 75, 75]),  # partial overlap
        ],
        ids=["perfect", "none", "partial"],
    )
    def test_shared_vs_gemex(self, box_a, box_b) -> None:
        from examples.gemex_thinkvg.src.rewards.bbox import compute_iou as gemex_iou
        from gaze.utils.iou import compute_iou as shared_iou

        shared = shared_iou([float(x) for x in box_a], [float(x) for x in box_b])
        gemex = gemex_iou(box_a, box_b)
        assert gemex == pytest.approx(shared)

    def test_gemex_normalizes_reversed_coords_the_shared_helper_rejects(self) -> None:
        """GEMeX clamps (and so reorders) before scoring; the shared default does not."""
        from examples.gemex_thinkvg.src.rewards.bbox import compute_iou as gemex_iou
        from gaze.utils.iou import compute_iou as shared_iou

        reversed_box, normal_box = [100, 50, 20, 80], [20, 50, 100, 80]

        assert shared_iou([float(x) for x in reversed_box], [float(x) for x in normal_box]) == 0.0
        assert gemex_iou(reversed_box, normal_box) > 0.9


# ===========================================================================
# _normalize_diagnosis parity (NOVA example vs env)
# ===========================================================================


class TestNormalizeDiagnosisParity:
    """NOVA example rewards and environment rewards _normalize_diagnosis must agree.

    Note: normalize_diagnosis_string in evaluation/diagnosis.py is intentionally
    different — it does NOT strip hedging modifiers, as it's used for exact
    string matching in evaluation. The reward normalizers DO strip hedging
    for more lenient RL training signal.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Glioblastoma",
            "possible meningioma",
            "mild hydrocephalus",
            "GBM grade IV",
            "septo-optic dysplasia",
            "ct scan findings",
            "Dandy-Walker malformation (variant)",
        ],
        ids=["simple", "hedging", "severity", "abbreviation", "hyphenated", "abbrev_ct", "parens"],
    )
    def test_nova_reward_vs_env_reward(self, text) -> None:
        sys.path.insert(0, str(EXAMPLE_ROOT))
        sys.path.insert(0, str(ENV_SRC))
        try:
            from nova_brain_mri.rewards import _normalize_diagnosis as env_norm
            from src.rewards import _normalize_diagnosis as reward_norm
        finally:
            sys.path.pop(0)
            sys.path.pop(0)

        assert reward_norm(text) == env_norm(text), (
            f"Divergence for {text!r}: reward={reward_norm(text)!r} vs env={env_norm(text)!r}"
        )


# ===========================================================================
# _ABBREVIATION_MAPPING parity
# ===========================================================================


class TestAbbreviationMappingParity:
    """All _ABBREVIATION_MAPPING instances must be identical."""

    def test_nova_caption_stopwords_vs_env(self) -> None:
        """The env keeps its own copy of the stopword list; it must not drift."""
        sys.path.insert(0, str(EXAMPLE_ROOT))
        sys.path.insert(0, str(ENV_SRC))
        try:
            from nova_brain_mri.rewards import _CAPTION_STOPWORDS as env_words
            from src.rewards import _CAPTION_STOPWORDS as reward_words
        finally:
            sys.path.pop(0)
            sys.path.pop(0)

        assert reward_words == env_words

    def test_nova_reward_vs_env(self) -> None:
        sys.path.insert(0, str(EXAMPLE_ROOT))
        sys.path.insert(0, str(ENV_SRC))
        try:
            from nova_brain_mri.rewards import _ABBREVIATION_MAPPING as env_map
            from src.rewards import _ABBREVIATION_MAPPING as reward_map
        finally:
            sys.path.pop(0)
            sys.path.pop(0)

        assert reward_map == env_map


# ===========================================================================
# CombinedReward negative weight validation
# ===========================================================================


class TestCombinedRewardNegativeWeights:
    """CombinedReward must reject negative weights."""

    def test_raises_on_negative_weight(self) -> None:
        from gaze.verifiers.rewards import CombinedReward
        from gaze.verifiers.rewards import ExactMatchReward
        from gaze.verifiers.rewards import TokenF1Reward

        with pytest.raises(ValueError, match="non-negative"):
            CombinedReward(
                rewards=[ExactMatchReward(), TokenF1Reward()],
                weights=[-0.5, 1.5],
            )


class TestEmptyGroundTruthEarnsNoCredit:
    """A missing diagnosis label must not hand out free reward.

    ``_normalize_diagnosis("")`` returns ``""``, so an empty prediction used to
    match an empty reference exactly: top-1 hit plus full coverage = 1.0. On a
    ``task="all"`` rubric that is 0.34 of the weighted score for saying nothing.
    """

    @staticmethod
    def _nova_reward():
        sys.path.insert(0, str(EXAMPLE_ROOT))
        try:
            from src.rewards import compute_diagnosis_reward

            return compute_diagnosis_reward
        finally:
            sys.path.pop(0)

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [("", ""), ("", "glioblastoma"), ("glioblastoma", "")],
        ids=["both_empty", "empty_prediction", "empty_reference"],
    )
    def test_empty_side_scores_zero(self, prediction: str, reference: str) -> None:
        assert self._nova_reward()(prediction, reference) == 0.0

    def test_real_matches_are_unaffected(self) -> None:
        reward = self._nova_reward()
        assert reward("glioblastoma", "glioblastoma") == 1.0
        assert reward("GBM", "glioblastoma multiforme") == 1.0
        assert reward("meningioma", "glioblastoma") == 0.0

    def test_example_and_env_agree_on_empty_ground_truth(self) -> None:
        """The env already guarded this; the example must not drift back."""
        sys.path.insert(0, str(EXAMPLE_ROOT))
        sys.path.insert(0, str(ENV_SRC))
        try:
            from nova_brain_mri.rewards import diagnosis_reward as env_reward
            from src.rewards import compute_diagnosis_reward as example_reward
        finally:
            sys.path.pop(0)
            sys.path.pop(0)

        empty_completion = [{"role": "assistant", "content": ""}]
        assert example_reward("", "") == 0.0
        assert env_reward("", empty_completion, {"diagnosis": ""}) == 0.0
