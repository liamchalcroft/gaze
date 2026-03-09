"""Tests for AgentClinic NEJM multi-turn environment.

Validates verifiers API conformance, state management, stop conditions,
reward calculation, and helper functions.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import verifiers as vf

from examples.agentclinic_nejm.src.environment import AgentClinicNEJMMultiTurn
from examples.agentclinic_nejm.src.environment import _brace_content
from examples.agentclinic_nejm.src.environment import _normalize
from examples.agentclinic_nejm.src.environment import accuracy_reward
from examples.agentclinic_nejm.src.environment import combined_reward

# ---------- Fixtures ----------
CASE_PNEUMONIA: dict[str, Any] = {
    "question": "Patient presents with cough and fever",
    "patient_info": "3-week history of productive cough",
    "physical_exams": "Bilateral lung crackles on auscultation",
    "answers": [
        {"text": "Pneumonia", "correct": True},
        {"text": "Bronchitis", "correct": False},
    ],
    "image_url": "",
}

CASE_WITH_IMAGE: dict[str, Any] = {
    "question": "Skin lesion on forearm",
    "patient_info": "6-month slowly growing lesion",
    "physical_exams": "Well-circumscribed nodule",
    "answers": [
        {"text": "Melanoma", "correct": False},
        {"text": "Dermatofibroma", "correct": True},
    ],
    "image_url": "https://example.com/image.jpg",
}


def _make_env(
    cases: list[dict[str, Any]] | None = None, max_turns: int = 5
) -> AgentClinicNEJMMultiTurn:
    return AgentClinicNEJMMultiTurn(cases=cases or [CASE_PNEUMONIA], max_turns=max_turns)


def _make_state(**overrides: Any) -> vf.State:
    state = vf.State()
    state["asked"] = False
    state["trajectory"] = []
    state["info"] = {
        "gold": "Pneumonia",
        "answers": [
            {"text": "Pneumonia", "correct": True},
            {"text": "Bronchitis", "correct": False},
        ],
        "patient_info": "3-week history of productive cough",
        "physical_exams": "Bilateral lung crackles on auscultation",
        "image_url": "",
    }
    for k, v in overrides.items():
        state[k] = v
    return state


# ---------- 1. Signature conformance ----------
class TestSignatureConformance:
    def test_env_response_signature_matches_base(self):
        base_sig = inspect.signature(vf.MultiTurnEnv.env_response)
        sub_sig = inspect.signature(AgentClinicNEJMMultiTurn.env_response)
        base_params = set(base_sig.parameters) - {"self"}
        sub_params = set(sub_sig.parameters) - {"self"}
        assert base_params <= sub_params, (
            f"env_response missing base params: {base_params - sub_params}"
        )

    def test_does_not_override_is_completed(self):
        assert "is_completed" not in AgentClinicNEJMMultiTurn.__dict__, (
            "is_completed should not be overridden; use @vf.stop methods instead"
        )

    def test_diagnosis_given_is_stop_condition(self):
        assert hasattr(AgentClinicNEJMMultiTurn.diagnosis_given, "stop"), (
            "diagnosis_given must be decorated with @vf.stop"
        )

    def test_setup_state_is_overridden(self):
        assert "setup_state" in AgentClinicNEJMMultiTurn.__dict__


# ---------- 2. Constructor ----------
class TestConstructor:
    def test_max_turns_passed_to_base(self):
        env = _make_env(max_turns=7)
        assert env.max_turns == 7

    def test_rubric_is_set(self):
        env = _make_env()
        assert env.rubric is not None
        assert len(env.rubric.funcs) == 1

    def test_dataset_has_expected_columns(self):
        env = _make_env()
        cols = set(env.dataset.column_names)
        assert "prompt" in cols
        assert "info" in cols


# ---------- 3. setup_state ----------
class TestSetupState:
    def test_initializes_asked(self):
        env = _make_env()
        state = vf.State()
        result = asyncio.run(env.setup_state(state))
        assert result["asked"] is False

    def test_returns_same_state_object(self):
        env = _make_env()
        state = vf.State()
        result = asyncio.run(env.setup_state(state))
        assert result is state


# ---------- 4. env_response ----------
class TestEnvResponse:
    def test_returns_list_of_messages(self):
        env = _make_env()
        state = _make_state()
        messages: vf.Messages = [
            {"role": "assistant", "content": "Tell me about the patient's HISTORY"},
        ]
        result = asyncio.run(env.env_response(messages, state))
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_history_keyword_sets_asked(self):
        env = _make_env()
        state = _make_state()
        messages: vf.Messages = [
            {"role": "assistant", "content": "What is the patient's medical HISTORY?"},
        ]
        result = asyncio.run(env.env_response(messages, state))
        assert state["asked"] is True
        assert "3-week history" in result[0]["content"]

    def test_symptom_keyword_sets_asked(self):
        env = _make_env()
        state = _make_state()
        messages: vf.Messages = [
            {"role": "assistant", "content": "What symptoms does the patient have?"},
        ]
        asyncio.run(env.env_response(messages, state))
        assert state["asked"] is True

    def test_exam_keyword_sets_asked(self):
        env = _make_env()
        state = _make_state()
        messages: vf.Messages = [
            {"role": "assistant", "content": "Show me the physical examination findings"},
        ]
        result = asyncio.run(env.env_response(messages, state))
        assert state["asked"] is True
        assert "crackles" in result[0]["content"]

    def test_lab_keyword_sets_asked(self):
        env = _make_env()
        state = _make_state()
        messages: vf.Messages = [
            {"role": "assistant", "content": "Are there lab results?"},
        ]
        asyncio.run(env.env_response(messages, state))
        assert state["asked"] is True

    def test_image_keyword_no_image(self):
        env = _make_env()
        state = _make_state()
        messages: vf.Messages = [
            {"role": "assistant", "content": "Can I see the medical image?"},
        ]
        result = asyncio.run(env.env_response(messages, state))
        assert state["asked"] is True
        assert "No medical image" in result[0]["content"]

    def test_image_keyword_with_image(self):
        env = _make_env(cases=[CASE_WITH_IMAGE])
        state = _make_state(
            info={
                **CASE_WITH_IMAGE,
                "gold": "Dermatofibroma",
            }
        )
        messages: vf.Messages = [
            {"role": "assistant", "content": "Show me the image"},
        ]
        result = asyncio.run(env.env_response(messages, state))
        assert state["asked"] is True
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[1]["type"] == "image_url"

    def test_no_keyword_sends_nudge(self):
        env = _make_env()
        state = _make_state()
        messages: vf.Messages = [
            {"role": "assistant", "content": "Hmm, let me think about this case."},
        ]
        result = asyncio.run(env.env_response(messages, state))
        assert state["asked"] is False
        assert "HISTORY" in result[0]["content"]

    def test_mutates_state_directly(self):
        env = _make_env()
        state = _make_state()
        messages: vf.Messages = [
            {"role": "assistant", "content": "Tell me the HISTORY"},
        ]
        asyncio.run(env.env_response(messages, state))
        assert state["asked"] is True


# ---------- 5. diagnosis_given stop condition ----------
class TestDiagnosisGiven:
    def test_fires_when_asked_and_braces(self):
        env = _make_env()
        state = _make_state(
            asked=True,
            trajectory=[{"completion": [{"role": "assistant", "content": "{Pneumonia}"}]}],
        )
        assert asyncio.run(env.diagnosis_given(state)) is True

    def test_blocked_when_not_asked(self):
        env = _make_env()
        state = _make_state(
            asked=False,
            trajectory=[{"completion": [{"role": "assistant", "content": "{Pneumonia}"}]}],
        )
        assert asyncio.run(env.diagnosis_given(state)) is False

    def test_blocked_when_no_braces(self):
        env = _make_env()
        state = _make_state(
            asked=True,
            trajectory=[{"completion": [{"role": "assistant", "content": "I think Pneumonia"}]}],
        )
        assert asyncio.run(env.diagnosis_given(state)) is False

    def test_blocked_when_no_trajectory(self):
        env = _make_env()
        state = _make_state(asked=True, trajectory=[])
        assert asyncio.run(env.diagnosis_given(state)) is False

    def test_blocked_when_empty_completion(self):
        env = _make_env()
        state = _make_state(
            asked=True,
            trajectory=[{"completion": []}],
        )
        assert asyncio.run(env.diagnosis_given(state)) is False


# ---------- 6. Reward functions ----------
class TestRewards:
    def test_accuracy_correct(self):
        info = {
            "gold": "Pneumonia",
            "answers": [
                {"text": "Pneumonia", "correct": True},
                {"text": "Bronchitis", "correct": False},
            ],
        }
        completion = [{"role": "assistant", "content": "{Pneumonia}"}]
        assert accuracy_reward("", completion, info) == 1.0

    def test_accuracy_incorrect(self):
        info = {
            "gold": "Pneumonia",
            "answers": [
                {"text": "Pneumonia", "correct": True},
                {"text": "Bronchitis", "correct": False},
            ],
        }
        completion = [{"role": "assistant", "content": "{Bronchitis}"}]
        assert accuracy_reward("", completion, info) == 0.0

    def test_accuracy_case_insensitive(self):
        info = {"gold": "Pneumonia", "answers": []}
        completion = [{"role": "assistant", "content": "{pneumonia}"}]
        assert accuracy_reward("", completion, info) == 1.0

    def test_accuracy_no_braces(self):
        info = {"gold": "Pneumonia", "answers": []}
        completion = [{"role": "assistant", "content": "Pneumonia"}]
        assert accuracy_reward("", completion, info) == 1.0

    def test_accuracy_empty_gold_uses_answers(self):
        info = {
            "gold": "",
            "answers": [
                {"text": "Pneumonia", "correct": True},
                {"text": "Bronchitis", "correct": False},
            ],
        }
        completion = [{"role": "assistant", "content": "{Pneumonia}"}]
        assert accuracy_reward("", completion, info) == 1.0

    def test_accuracy_fallback_correct_option(self):
        info = {
            "gold": "Something else",
            "answers": [
                {"text": "Pneumonia", "correct": True},
            ],
        }
        completion = [{"role": "assistant", "content": "{Pneumonia}"}]
        assert accuracy_reward("", completion, info) == 1.0

    def test_combined_reward_range(self):
        info = {"gold": "Pneumonia", "answers": []}
        completion = [{"role": "assistant", "content": "{Pneumonia}"}]
        r = combined_reward("", completion, info)
        assert 0.0 <= r <= 1.0

    def test_combined_reward_correct_is_high(self):
        info = {"gold": "Pneumonia", "answers": []}
        completion = [{"role": "assistant", "content": "{Pneumonia}"}]
        assert combined_reward("", completion, info) > 0.8


# ---------- 7. Helper functions ----------
class TestHelpers:
    def test_brace_content_simple(self):
        assert _brace_content("The answer is {Pneumonia}.") == "Pneumonia"

    def test_brace_content_empty(self):
        assert _brace_content("No braces here") == ""

    def test_brace_content_multiple_picks_last(self):
        assert _brace_content("{first} then {second}") == "second"

    def test_brace_content_with_options_picks_match(self):
        assert (
            _brace_content("{Diagnosis} is {Pneumonia}", ["Pneumonia", "Bronchitis"]) == "Pneumonia"
        )

    def test_brace_content_with_options_rejects_placeholder(self):
        assert _brace_content("{Unknown}", ["Pneumonia", "Bronchitis"]) == ""

    def test_normalize_lowercase(self):
        assert _normalize("Pneumonia") == "pneumonia"

    def test_normalize_strips_braces(self):
        assert _normalize("{Pneumonia}") == "pneumonia"

    def test_normalize_strips_parens_and_dots(self):
        assert _normalize("(A) Pneumonia.") == "a) pneumonia"

    def test_normalize_empty(self):
        assert _normalize("") == ""


# ---------- 8. Dataset round-trip ----------
class TestDatasetRoundTrip:
    def test_info_survives_dataset(self):
        env = _make_env()
        row = env.dataset[0]
        info = row["info"]
        if isinstance(info, str):
            import json

            info = json.loads(info)
        assert info["gold"] == "Pneumonia"
        assert len(info["answers"]) == 2

    def test_prompt_has_system_and_user(self):
        env = _make_env()
        row = env.dataset[0]
        prompt = row["prompt"]
        roles = [m["role"] for m in prompt]
        assert "system" in roles
        assert "user" in roles
