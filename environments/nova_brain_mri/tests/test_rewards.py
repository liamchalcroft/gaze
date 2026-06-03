"""Tests for NOVA Brain MRI environment reward functions."""

from __future__ import annotations

import json
from typing import Any

import pytest


def test_no_gaze_imports():
    """rewards.py must not import from gaze."""
    from pathlib import Path

    rewards_path = Path(__file__).parent.parent / "src" / "nova_brain_mri" / "rewards.py"
    source = rewards_path.read_text()
    assert "from gaze" not in source
    assert "import gaze" not in source


def test_depends_on_gaze_in_pyproject():
    """The environment depends on the GAZE framework (gaze-vlm)."""
    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    source = pyproject_path.read_text()
    assert "gaze-vlm" in source


def test_utils_reexports_from_gaze():
    """_utils.py wires the extraction/IoU helpers to GAZE."""
    from pathlib import Path

    utils_path = Path(__file__).parent.parent / "src" / "nova_brain_mri" / "_utils.py"
    source = utils_path.read_text()
    assert "from gaze.utils import" in source
    assert "from gaze.verifiers.rewards import" in source


def test_utils_importable():
    from nova_brain_mri._utils import (
        compute_iou,
        extract_completion_text,
        extract_json_from_text,
    )

    assert callable(compute_iou)
    assert callable(extract_json_from_text)
    assert callable(extract_completion_text)


def _make_completion(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "assistant", "content": json.dumps(response)}]


class TestCaptionReward:
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
        """Counter-based: pred="mild" vs ref="mild mild" -> recall=0.5, F1=2/3."""
        from nova_brain_mri.rewards import caption_reward

        completion = _make_completion({"caption": "mild"})
        info = {"caption": "mild mild"}
        score = caption_reward("", completion, info)
        assert score < 0.9
        assert score == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_invalid_json_returns_zero(self):
        from nova_brain_mri.rewards import caption_reward

        completion = [{"role": "assistant", "content": "not json"}]
        info = {"caption": "something"}
        assert caption_reward("", completion, info) == 0.0


class TestDiagnosisNormalization:
    def test_preserves_hyphens(self):
        from nova_brain_mri.rewards import _normalize_diagnosis

        result = _normalize_diagnosis("septo-optic dysplasia")
        assert "-" in result

    def test_expands_abbreviation(self):
        from nova_brain_mri.rewards import _normalize_diagnosis

        assert "septo-optic dysplasia" in _normalize_diagnosis("SOD")

    def test_expands_avm(self):
        from nova_brain_mri.rewards import _normalize_diagnosis

        assert "arteriovenous malformation" in _normalize_diagnosis("AVM")

    def test_word_boundary_hedging(self):
        from nova_brain_mri.rewards import _normalize_diagnosis

        result_probable = _normalize_diagnosis("probable glioma")
        assert "probable" not in result_probable
        assert "glioma" in result_probable

        result_improbable = _normalize_diagnosis("improbable diagnosis")
        assert "improbable" in result_improbable

    def test_diagnosis_reward_abbreviation_match(self):
        from nova_brain_mri.rewards import diagnosis_reward

        completion = _make_completion(
            {"diagnosis": {"primary_diagnosis": "AVM", "differential_diagnoses": []}}
        )
        info = {"diagnosis": {"primary": "arteriovenous malformation"}}
        assert diagnosis_reward("", completion, info) == pytest.approx(1.0)


class TestLocalizationReward:
    def test_perfect_iou(self):
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory(iou_threshold=0.5)
        completion = _make_completion({"localization": [{"bounding_box": [10, 10, 50, 50]}]})
        info = {"boxes": [[10, 10, 50, 50]]}
        assert loc_reward("", completion, info) == pytest.approx(1.0)

    def test_rejects_bbox_key_from_predictions(self):
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory(iou_threshold=0.5)
        completion = _make_completion({"localization": [{"bbox": [10, 10, 50, 50]}]})
        info = {"boxes": [[10, 10, 50, 50]]}
        assert loc_reward("", completion, info) == 0.0

    def test_area_penalty_full_image_box(self):
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory(iou_threshold=0.1)
        completion = _make_completion({"localization": [{"bounding_box": [0, 0, 100, 100]}]})
        info = {
            "boxes": [[20, 20, 40, 40]],
            "image_width": 100,
            "image_height": 100,
        }
        assert loc_reward("", completion, info) < 0.1

    def test_no_area_penalty_without_dimensions(self):
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory(iou_threshold=0.5)
        completion = _make_completion({"localization": [{"bounding_box": [10, 10, 50, 50]}]})
        info = {"boxes": [[10, 10, 50, 50]]}
        assert loc_reward("", completion, info) == pytest.approx(1.0)

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
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory()
        completion = _make_completion({"localization": [{"bounding_box": [0, 0, 50, 50]}]})
        info = {"boxes": [[30, 30, 80, 80]]}
        # IoU = 400/4600 ~ 0.087 < 0.5
        assert loc_reward("", completion, info) == 0.0


class TestConfigDefaults:
    def test_env_config_iou_default(self):
        from nova_brain_mri import NOVAEnvConfig

        config = NOVAEnvConfig()
        assert config.iou_threshold == 0.5

    def test_load_iou_default(self):
        import inspect

        from nova_brain_mri import load

        sig = inspect.signature(load)
        assert sig.parameters["iou_threshold"].default == 0.5


class TestSystemPromptNoPhantomTools:
    def _get_prompt(self, **config_kwargs: Any) -> str:
        from nova_brain_mri import NOVABrainMRIEnv, NOVAEnvConfig

        env = object.__new__(NOVABrainMRIEnv)
        env.config = NOVAEnvConfig(**config_kwargs)
        return env._get_system_prompt()

    def test_no_tool_names_in_prompt(self):
        prompt = self._get_prompt()
        for name in ["zoom", "crop", "adjust_contrast", "threshold", "reset"]:
            assert name not in prompt

    def test_no_search_names_in_prompt(self):
        prompt = self._get_prompt()
        for name in ["search_web", "search_images", "PubMed"]:
            assert name not in prompt

    def test_no_use_tools_instruction(self):
        prompt = self._get_prompt()
        assert "Use tools" not in prompt


class TestStateCleanliness:
    def test_no_tool_uses_in_state(self):
        import asyncio

        from nova_brain_mri import NOVABrainMRIEnv, NOVAEnvConfig

        env = object.__new__(NOVABrainMRIEnv)
        env.config = NOVAEnvConfig()
        state = asyncio.run(env.setup_state({}))
        assert "tool_uses" not in state


class TestCLISchemaFlag:
    def test_schema_flag_in_parser(self):
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
        import sys
        from unittest.mock import patch

        from nova_brain_mri.cli import parse_args

        with patch.object(sys, "argv", ["cli", "--schema"]):
            args = parse_args()
        assert args.schema is True
        assert args.model is None


class TestUtilsParity:
    def test_compute_iou_basic(self):
        from nova_brain_mri._utils import compute_iou

        assert compute_iou([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]) == pytest.approx(1.0)
        assert compute_iou([0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]) == pytest.approx(0.0)

    def test_compute_iou_partial(self):
        from nova_brain_mri._utils import compute_iou

        iou = compute_iou([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0])
        assert iou == pytest.approx(50 / 150)

    def test_extract_json_from_text_basic(self):
        from nova_brain_mri._utils import extract_json_from_text

        assert extract_json_from_text('{"key": "value"}') == {"key": "value"}

    def test_extract_json_from_text_markdown(self):
        from nova_brain_mri._utils import extract_json_from_text

        assert extract_json_from_text('```json\n{"key": "value"}\n```') == {"key": "value"}

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

    def test_extract_completion_text_multimodal_concatenates_all(self):
        from nova_brain_mri._utils import extract_completion_text

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me think about this."},
                    {"type": "text", "text": '{"diagnosis": "glioma"}'},
                ],
            }
        ]
        result = extract_completion_text(messages)
        assert '{"diagnosis": "glioma"}' in result
        assert "Let me think" in result
        assert "\n" in result


class TestDiagnosisRewardPrimaryOnly:
    def test_correct_differential_scores_zero(self):
        from nova_brain_mri.rewards import diagnosis_reward

        completion = _make_completion(
            {
                "diagnosis": {
                    "primary_diagnosis": "wrong diagnosis",
                    "differential_diagnoses": [{"diagnosis": "glioma"}],
                }
            }
        )
        info = {"diagnosis": "glioma"}
        assert diagnosis_reward("", completion, info) == pytest.approx(0.0)

    def test_correct_primary_scores_one(self):
        from nova_brain_mri.rewards import diagnosis_reward

        completion = _make_completion(
            {
                "diagnosis": {
                    "primary_diagnosis": "glioma",
                    "differential_diagnoses": [{"diagnosis": "meningioma"}],
                }
            }
        )
        info = {"diagnosis": "glioma"}
        assert diagnosis_reward("", completion, info) == pytest.approx(1.0)


class TestIoUCrossImplementation:
    _BOX_PAIRS = [
        ([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], 1.0),
        ([0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], 0.0),
        ([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], 50 / 150),
        ([0.0, 0.0, 100.0, 100.0], [50.0, 50.0, 150.0, 150.0], 2500 / 17500),
        ([10.0, 10.0, 50.0, 50.0], [10.0, 10.0, 50.0, 50.0], 1.0),
        ([0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0], 0.0),
        ([0.0, 0.0, 10.0, 10.0], [9.0, 9.0, 19.0, 19.0], 1 / 199),
        ([50.0, 50.0, 10.0, 10.0], [10.0, 10.0, 50.0, 50.0], 1.0),
        ([100.0, 100.0, 0.0, 0.0], [50.0, 50.0, 150.0, 150.0], 2500 / 17500),
    ]

    @pytest.mark.parametrize("box1,box2,expected", _BOX_PAIRS)
    def test_iou_values(self, box1, box2, expected):
        from nova_brain_mri._utils import compute_iou

        assert compute_iou(box1, box2) == pytest.approx(expected, abs=1e-6)


class TestNormalizeDiagnosisParity:
    _CASES = [
        ("SOD", "septo-optic dysplasia"),
        ("AVM", "arteriovenous malformation"),
        ("GBM", "glioblastoma multiforme"),
        ("MS", "multiple sclerosis"),
        ("NPH", "normal pressure hydrocephalus"),
        ("possible glioma", "glioma"),
        ("probable meningioma", "meningioma"),
        ("likely AVM", "arteriovenous malformation"),
        ("suspected MS", "multiple sclerosis"),
        ("septo-optic dysplasia", "septo-optic dysplasia"),
        ("improbable diagnosis", "improbable diagnosis"),
    ]

    @pytest.mark.parametrize("input_,expected", _CASES)
    def test_normalization(self, input_, expected):
        from nova_brain_mri.rewards import _normalize_diagnosis

        assert _normalize_diagnosis(input_) == expected


class TestEndToEndRewardParity:
    _COMPLETION = _make_completion(
        {
            "caption": "Axial T2 FLAIR showing periventricular white matter lesions",
            "diagnosis": {
                "primary_diagnosis": "multiple sclerosis",
                "differential_diagnoses": [{"diagnosis": "ADEM"}],
            },
            "localization": [{"bounding_box": [120, 80, 180, 140]}],
            "continue": False,
        }
    )

    def test_caption_reward_pinned(self):
        from nova_brain_mri.rewards import caption_reward

        info = {"caption": "Axial T2 FLAIR showing periventricular white matter lesions"}
        assert caption_reward("", self._COMPLETION, info) == pytest.approx(1.0)

    def test_caption_reward_partial(self):
        from nova_brain_mri.rewards import caption_reward

        info = {"caption": "Axial FLAIR showing some lesions"}
        score = caption_reward("", self._COMPLETION, info)
        assert 0.0 < score < 1.0

    def test_diagnosis_reward_pinned(self):
        from nova_brain_mri.rewards import diagnosis_reward

        info = {"diagnosis": "multiple sclerosis"}
        assert diagnosis_reward("", self._COMPLETION, info) == pytest.approx(1.0)

    def test_diagnosis_reward_ms_abbreviation(self):
        from nova_brain_mri.rewards import diagnosis_reward

        info = {"diagnosis": "MS"}
        assert diagnosis_reward("", self._COMPLETION, info) == pytest.approx(1.0)

    def test_localization_reward_pinned(self):
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory(iou_threshold=0.5)
        info = {"boxes": [[120, 80, 180, 140]]}
        assert loc_reward("", self._COMPLETION, info) == pytest.approx(1.0)

    def test_localization_reward_no_match(self):
        from nova_brain_mri.rewards import localization_reward_factory

        loc_reward = localization_reward_factory(iou_threshold=0.5)
        info = {"boxes": [[300, 300, 400, 400]]}
        assert loc_reward("", self._COMPLETION, info) == pytest.approx(0.0)
