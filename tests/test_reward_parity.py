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


def _env_extract_completion_text():
    """Import extract_completion_text from the standalone NOVA environment."""
    sys.path.insert(0, str(ENV_SRC))
    try:
        from nova_brain_mri._utils import extract_completion_text

        return extract_completion_text
    finally:
        sys.path.pop(0)


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


class TestExtractCompletionTextParity:
    """Core and standalone env must produce identical output."""

    @pytest.mark.parametrize(
        "completion",
        [_MULTIMODAL_COMPLETION, _STRING_COMPLETION, _EMPTY_COMPLETION, "plain string"],
        ids=["multimodal", "string_content", "no_assistant", "raw_string"],
    )
    def test_core_vs_env_parity(self, completion) -> None:
        from radiant_harness.verifiers.rewards import extract_completion_text as core_fn

        env_fn = _env_extract_completion_text()
        assert core_fn(completion) == env_fn(completion)


# ===========================================================================
# compute_iou parity
# ===========================================================================


class TestComputeIoUParity:
    """Shared compute_iou and GEMeX compute_iou must agree."""

    @pytest.mark.parametrize(
        "box_a,box_b",
        [
            ([0, 0, 100, 100], [0, 0, 100, 100]),  # perfect overlap
            ([0, 0, 10, 10], [200, 200, 300, 300]),  # no overlap
            ([100, 50, 20, 80], [20, 50, 100, 80]),  # reversed coords
            ([0, 0, 50, 50], [25, 25, 75, 75]),  # partial overlap
        ],
        ids=["perfect", "none", "reversed", "partial"],
    )
    def test_shared_vs_gemex(self, box_a, box_b) -> None:
        from examples.gemex_thinkvg.src.rewards.bbox import compute_iou as gemex_iou
        from radiant_harness.utils.iou import compute_iou as shared_iou

        shared = shared_iou([float(x) for x in box_a], [float(x) for x in box_b])
        gemex = gemex_iou(box_a, box_b)
        assert abs(shared - gemex) < 0.05, f"IoU mismatch: shared={shared:.4f} vs gemex={gemex:.4f}"


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

    def test_nova_reward_vs_diag(self) -> None:
        sys.path.insert(0, str(EXAMPLE_ROOT))
        try:
            from src.evaluation.diagnosis import _ABBREVIATION_MAPPING as diag_map
            from src.rewards import _ABBREVIATION_MAPPING as reward_map
        finally:
            sys.path.pop(0)

        assert reward_map == diag_map

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
        from radiant_harness.verifiers.rewards import CombinedReward
        from radiant_harness.verifiers.rewards import ExactMatchReward
        from radiant_harness.verifiers.rewards import TokenF1Reward

        with pytest.raises(ValueError, match="non-negative"):
            CombinedReward(
                rewards=[ExactMatchReward(), TokenF1Reward()],
                weights=[-0.5, 1.5],
            )
