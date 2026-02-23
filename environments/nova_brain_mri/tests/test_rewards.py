"""Tests for NOVA Brain MRI environment reward functions.

Validates:
- Independence from radiant_harness (no imports)
- Caption reward uses multiset intersection (Counter, not set)
- Diagnosis normalization preserves hyphens, expands abbreviations, uses word boundaries
- Localization reward enforces "bounding_box" key and applies area penalty
- IoU threshold defaults to 0.5
"""

from __future__ import annotations

import json
from typing import Any

import pytest


# ── Independence check ──────────────────────────────────────────────────────
def test_no_radiant_harness_imports():
    """rewards.py must not import from radiant_harness."""
    from pathlib import Path

    rewards_path = Path(__file__).parent.parent / "src" / "nova_brain_mri" / "rewards.py"
    source = rewards_path.read_text()
    assert "from radiant_harness" not in source
    assert "import radiant_harness" not in source


def test_no_radiant_harness_in_pyproject():
    """pyproject.toml must not list radiant-harness as a dependency."""
    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    source = pyproject_path.read_text()
    assert "radiant-harness" not in source


# ── Utility imports work from local _utils ──────────────────────────────────
def test_utils_importable():
    """_utils.py utilities should be importable from the package."""
    from nova_brain_mri._utils import (
        compute_iou,
        extract_completion_text,
        extract_json_from_text,
    )

    assert callable(compute_iou)
    assert callable(extract_json_from_text)
    assert callable(extract_completion_text)


# ── Caption reward ──────────────────────────────────────────────────────────
def _make_completion(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Helper: wrap a response dict as a verifiers-style completion."""
    return [{"role": "assistant", "content": json.dumps(response)}]


class TestCaptionReward:
    """Caption reward uses multiset (Counter) intersection."""

    def test_perfect_match(self):
        from nova_brain_mri.rewards import caption_reward

        completion = _make_completion({"caption": "mild white matter changes"})
        info = {"caption": "mild white matter changes"}
        assert caption_reward("", completion, info) == pytest.approx(1.0)

    def test_no_overlap(self):
        from nova_brain_mri.rewards import caption_reward

        completion = _make_completion({"caption": "normal brain"})
        info = {"caption": "severe hemorrhage detected"}
        assert caption_reward("", completion, info) == pytest.approx(0.0)

    def test_multiset_not_set(self):
        """Repeated tokens must affect the score (Counter, not set).

        If using set(), "mild" appearing once vs twice is ignored.
        With Counter: pred="mild" vs ref="mild mild" → recall=0.5, not 1.0.
        """
        from nova_brain_mri.rewards import caption_reward

        completion = _make_completion({"caption": "mild"})
        info = {"caption": "mild mild"}
        score = caption_reward("", completion, info)
        # With Counter: precision=1/1=1.0, recall=1/2=0.5, F1=2/3≈0.667
        # With set(): precision=1/1=1.0, recall=1/1=1.0, F1=1.0
        assert score < 0.9, f"Expected < 0.9 (Counter-based), got {score}"
        assert score == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_invalid_json_returns_zero(self):
        from nova_brain_mri.rewards import caption_reward

        completion = [{"role": "assistant", "content": "not json"}]
        info = {"caption": "something"}
        assert caption_reward("", completion, info) == 0.0


# ── Diagnosis normalization ─────────────────────────────────────────────────
class TestDiagnosisNormalization:
    """Normalization preserves hyphens, expands abbreviations, uses \\b."""

    def test_preserves_hyphens(self):
        from nova_brain_mri.rewards import _normalize_diagnosis

        result = _normalize_diagnosis("septo-optic dysplasia")
        assert "-" in result, f"Hyphens should be preserved, got: {result}"

    def test_expands_abbreviation(self):
        from nova_brain_mri.rewards import _normalize_diagnosis

        result = _normalize_diagnosis("SOD")
        assert "septo-optic dysplasia" in result

    def test_expands_avm(self):
        from nova_brain_mri.rewards import _normalize_diagnosis

        result = _normalize_diagnosis("AVM")
        assert "arteriovenous malformation" in result

    def test_word_boundary_hedging(self):
        """'probable' should be stripped but 'improbable' should not."""
        from nova_brain_mri.rewards import _normalize_diagnosis

        # "probable" alone should be removed
        result_probable = _normalize_diagnosis("probable glioma")
        assert "probable" not in result_probable
        assert "glioma" in result_probable

        # "improbable" should NOT be corrupted
        result_improbable = _normalize_diagnosis("improbable diagnosis")
        assert "improbable" in result_improbable

    def test_diagnosis_reward_abbreviation_match(self):
        """Abbreviation in prediction should match full term in reference."""
        from nova_brain_mri.rewards import diagnosis_reward

        completion = _make_completion(
            {"diagnosis": {"primary_diagnosis": "AVM", "differential_diagnoses": []}}
        )
        info = {"diagnosis": {"primary": "arteriovenous malformation"}}
        score = diagnosis_reward("", completion, info)
        # top1 match (0.6) + coverage 1/1 (0.4) = 1.0
        assert score == pytest.approx(1.0)


# ── Localization reward ─────────────────────────────────────────────────────
class TestLocalizationReward:
    """Localization reward enforces bounding_box key and applies area penalty."""

    def test_perfect_iou(self):
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory(iou_threshold=0.5)
        completion = _make_completion({"localization": [{"bounding_box": [10, 10, 50, 50]}]})
        info = {"boxes": [[10, 10, 50, 50]]}
        score = loc_reward("", completion, info)
        assert score == pytest.approx(1.0)

    def test_rejects_bbox_key_from_predictions(self):
        """Predictions using "bbox" instead of "bounding_box" should be ignored."""
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory(iou_threshold=0.5)
        # Use "bbox" key (ground-truth convention, not valid for predictions)
        completion = _make_completion({"localization": [{"bbox": [10, 10, 50, 50]}]})
        info = {"boxes": [[10, 10, 50, 50]]}
        score = loc_reward("", completion, info)
        # Should get 0.0 because no valid pred boxes extracted
        assert score == 0.0

    def test_area_penalty_full_image_box(self):
        """A box covering the full image should be penalized toward 0."""
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory(iou_threshold=0.1)
        # Pred box covers 100% of image (0,0 to 100,100 in a 100x100 image)
        completion = _make_completion({"localization": [{"bounding_box": [0, 0, 100, 100]}]})
        # Ref box is small (should have high IoU with full-image box)
        info = {
            "boxes": [[20, 20, 40, 40]],
            "image_width": 100,
            "image_height": 100,
        }
        score = loc_reward("", completion, info)
        # With area penalty: box covers 100% → penalty = 0.0, so score ≈ 0.0
        assert score < 0.1, f"Full-image box should be penalized, got {score}"

    def test_no_area_penalty_without_dimensions(self):
        """Without image dimensions, area penalty should not apply."""
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory(iou_threshold=0.5)
        completion = _make_completion({"localization": [{"bounding_box": [10, 10, 50, 50]}]})
        info = {"boxes": [[10, 10, 50, 50]]}  # No image_width/height
        score = loc_reward("", completion, info)
        assert score == pytest.approx(1.0)

    def test_both_empty_returns_one(self):
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory()
        completion = _make_completion({"localization": []})
        info = {"boxes": []}
        assert loc_reward("", completion, info) == 1.0

    def test_pred_empty_ref_present_returns_zero(self):
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory()
        completion = _make_completion({"localization": []})
        info = {"boxes": [[10, 10, 50, 50]]}
        assert loc_reward("", completion, info) == 0.0

    def test_default_iou_threshold_is_05(self):
        """Factory default should be 0.5, matching NOVA ACC50."""
        from nova_brain_mri.rewards import localization_reward_factory

        # A match with IoU=0.4 should fail at default threshold (0.5)
        loc_reward = localization_reward_factory()
        # Two boxes with partial overlap: IoU ≈ 0.39
        completion = _make_completion({"localization": [{"bounding_box": [0, 0, 50, 50]}]})
        info = {"boxes": [[30, 30, 80, 80]]}
        score = loc_reward("", completion, info)
        # IoU of [0,0,50,50] vs [30,30,80,80]:
        # intersection: [30,30,50,50] = 20*20 = 400
        # union: 2500 + 2500 - 400 = 4600
        # IoU = 400/4600 ≈ 0.087 < 0.5, so score = 0.0
        assert score == 0.0


# ── Config defaults ─────────────────────────────────────────────────────────
class TestConfigDefaults:
    def test_env_config_iou_default(self):
        from nova_brain_mri import NOVAEnvConfig

        config = NOVAEnvConfig()
        assert config.iou_threshold == 0.5

    def test_load_iou_default(self):
        """load() signature should default to 0.5."""
        import inspect

        from nova_brain_mri import load

        sig = inspect.signature(load)
        iou_param = sig.parameters["iou_threshold"]
        assert iou_param.default == 0.5


# ── System prompt: no phantom tool descriptions (#8) ────────────────────────
class TestSystemPromptNoPhantomTools:
    """System prompt must not describe tools that env_response cannot execute."""

    def _get_prompt(self, **config_kwargs: Any) -> str:
        """Build a system prompt via NOVABrainMRIEnv._get_system_prompt."""
        from nova_brain_mri import NOVAEnvConfig

        # Instantiate config without creating the full env (avoids dataset load)
        from nova_brain_mri import NOVABrainMRIEnv

        env = object.__new__(NOVABrainMRIEnv)
        env.config = NOVAEnvConfig(**config_kwargs)
        return env._get_system_prompt()

    def test_no_tool_names_in_prompt_with_tools_enabled(self):
        prompt = self._get_prompt(use_tools=True)
        for tool_name in ["zoom", "crop", "adjust_contrast", "threshold", "reset"]:
            assert tool_name not in prompt, (
                f"System prompt mentions '{tool_name}' but env_response does not execute tools"
            )

    def test_no_search_names_in_prompt_with_search_enabled(self):
        prompt = self._get_prompt(use_web_search=True)
        for name in ["search_web", "search_images", "PubMed"]:
            assert name not in prompt, (
                f"System prompt mentions '{name}' but env_response does not execute search"
            )

    def test_no_use_tools_instruction(self):
        """Final instruction should not say 'Use tools'."""
        prompt = self._get_prompt(use_tools=True, use_web_search=True)
        assert "Use tools" not in prompt


# ── State: no dead tool_uses counter (#8) ───────────────────────────────────
class TestStateCleanliness:
    """build_initial_state should not contain dead counters."""

    def test_no_tool_uses_in_state(self):
        from nova_brain_mri import NOVABrainMRIEnv, NOVAEnvConfig

        env = object.__new__(NOVABrainMRIEnv)
        env.config = NOVAEnvConfig()
        state = env.build_initial_state(
            prompt=[{"role": "user", "content": "test"}],
            info={"case_index": 0, "task": "all"},
        )
        assert "tool_uses" not in state


# ── CLI --schema flag (#9) ──────────────────────────────────────────────────
class TestCLISchemaFlag:
    """--schema flag should be wired up and reachable."""

    def test_schema_flag_in_parser(self):
        """parse_args should accept --schema without error."""
        import sys
        from unittest.mock import patch

        from nova_brain_mri.cli import parse_args

        with patch.object(sys, "argv", ["cli", "--schema", "-m", "dummy"]):
            args = parse_args()
        assert args.schema is True

    def test_schema_prints_json(self, capsys: pytest.CaptureFixture[str]):
        from nova_brain_mri.cli import print_env_schema

        print_env_schema()
        output = capsys.readouterr().out
        schema = json.loads(output)
        assert "properties" in schema
        assert "iou_threshold" in schema["properties"]
        assert schema["properties"]["iou_threshold"]["default"] == 0.5

    def test_model_not_required_with_schema(self):
        """--schema should work without --model."""
        import sys
        from unittest.mock import patch

        from nova_brain_mri.cli import parse_args

        with patch.object(sys, "argv", ["cli", "--schema"]):
            args = parse_args()
        assert args.schema is True
        assert args.model is None


# ── _utils parity ──────────────────────────────────────────────────────────
class TestUtilsParity:
    """Inlined _utils functions produce same results as radiant_harness originals."""

    def test_compute_iou_basic(self):
        from nova_brain_mri._utils import compute_iou

        assert compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
        assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)

    def test_compute_iou_partial(self):
        from nova_brain_mri._utils import compute_iou

        # 50% overlap
        iou = compute_iou([0, 0, 10, 10], [5, 0, 15, 10])
        # intersection: [5,0,10,10] = 50, union: 100+100-50=150
        assert iou == pytest.approx(50 / 150)

    def test_extract_json_from_text_basic(self):
        from nova_brain_mri._utils import extract_json_from_text

        result = extract_json_from_text('{"key": "value"}')
        assert result == {"key": "value"}

    def test_extract_json_from_text_markdown(self):
        from nova_brain_mri._utils import extract_json_from_text

        result = extract_json_from_text('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_extract_json_from_text_embedded(self):
        from nova_brain_mri._utils import extract_json_from_text

        result = extract_json_from_text('Here is my answer: {"diagnosis": "glioma"} done.')
        assert result == {"diagnosis": "glioma"}

    def test_extract_completion_text_string(self):
        from nova_brain_mri._utils import extract_completion_text

        assert extract_completion_text("hello") == "hello"

    def test_extract_completion_text_messages(self):
        from nova_brain_mri._utils import extract_completion_text

        messages = [
            {"role": "user", "content": "analyze"},
            {"role": "assistant", "content": "result text"},
        ]
        assert extract_completion_text(messages) == "result text"
