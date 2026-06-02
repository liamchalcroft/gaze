"""Hermetic smoke tests for the AgentClinic NEJM example.

No network, no dataset file, no model, no API keys. Exercises the
diagnosis-extraction and normalisation helpers, the accuracy reward on
synthetic completions, and (when verifiers is installed) builds a tiny
in-memory environment from synthetic cases to confirm the multi-turn
plumbing wires up.

The environment module imports ``verifiers`` and ``datasets`` at import
time, so the whole module is skipped cleanly when those optional
dependencies are absent.
"""

from __future__ import annotations

import pytest

# The environment module imports verifiers + datasets at module scope.
pytest.importorskip("verifiers")
pytest.importorskip("datasets")

from examples.agentclinic_nejm.src import environment as env_mod  # noqa: E402

_ANSWERS = [
    {"text": "Glioblastoma", "correct": True},
    {"text": "Meningioma", "correct": False},
    {"text": "Metastasis", "correct": False},
]


def test_brace_content_prefers_known_option() -> None:
    """When answer options are supplied, the matching brace value is returned."""
    text = "Considering {Meningioma} initially, the final answer is {Glioblastoma}."
    options = [a["text"] for a in _ANSWERS]
    assert env_mod._brace_content(text, options) == "Glioblastoma"
    # Without options, the last brace match is returned.
    assert env_mod._brace_content("first {A} then {B}") == "B"
    # No braces -> empty string.
    assert env_mod._brace_content("no braces here") == ""


def test_normalize_strips_punctuation_and_case() -> None:
    """Normalisation lowercases and strips surrounding braces/punctuation."""
    assert env_mod._normalize("  {Glioblastoma}.  ") == "glioblastoma"
    assert env_mod._normalize("") == ""


def test_extract_gold_finds_correct_answer() -> None:
    """The gold answer is the option flagged correct (bool or 'true' string)."""
    assert env_mod._extract_gold(_ANSWERS) == "Glioblastoma"
    string_flagged = [{"text": "X", "correct": "false"}, {"text": "Y", "correct": "true"}]
    assert env_mod._extract_gold(string_flagged) == "Y"
    assert env_mod._extract_gold([]) == ""


def test_accuracy_reward_correct_and_incorrect() -> None:
    """The accuracy reward is 1.0 for the gold diagnosis and 0.0 otherwise."""
    info = {"gold": "Glioblastoma", "answers": _ANSWERS}
    assert env_mod.accuracy_reward("", "My diagnosis is {Glioblastoma}", info) == pytest.approx(1.0)
    assert env_mod.accuracy_reward("", "My diagnosis is {Meningioma}", info) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_environment_from_synthetic_cases() -> None:
    """A tiny in-memory environment builds and its stop/response logic works."""
    cases = [
        {
            "question": "Headache and visual disturbance",
            "patient_info": "45-year-old with progressive headache.",
            "physical_exams": "Papilloedema on fundoscopy.",
            "answers": _ANSWERS,
            "image_url": "",
            "type": ["neurology"],
        }
    ]
    env = env_mod.AgentClinicNEJMMultiTurn(cases=cases, max_turns=4)
    assert len(env.dataset) == 1

    # A request for history triggers the patient-history branch and flips "asked".
    state = await env.setup_state({"info": env.dataset["info"][0]})
    messages = [{"role": "assistant", "content": "Can you give me the patient HISTORY?"}]
    reply = await env.env_response(messages, state)
    assert state["asked"] is True
    assert "Patient History" in reply[0]["content"]


def test_load_environment_rejects_missing_dataset(tmp_path) -> None:
    """Loading from a non-existent dataset path raises rather than hanging."""
    missing = tmp_path / "does_not_exist.jsonl"
    with pytest.raises((FileNotFoundError, OSError)):
        env_mod.load_environment(dataset_path=str(missing))
