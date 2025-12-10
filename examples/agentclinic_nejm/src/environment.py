"""AgentClinic NEJM multi-turn evaluation environment.

- Assistant asks; env replies with HISTORY / EXAM / TESTS / IMAGE
- Episode completes when assistant provides diagnosis in {Diagnosis} braces
  AND has made at least one information request
- Reward = 1 if normalized prediction matches gold answer

Provides verifiers-compatible environment with accuracy reward.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import verifiers as vf
from datasets import Dataset

from radiant_harness.verifiers import BaseRewardFunction

# ---------- Config ----------
DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "../data/agentclinic_nejm_extended.jsonl",
)

SYSTEM_PROMPT = (
    "Think step by step. When you are ready, output ONE diagnosis inside {Diagnosis} "
    "with no extra words and no punctuation inside the braces."
)

LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
LOG_PATH = os.path.join(LOG_DIR, "debug.log")


# ---------- Helpers ----------
def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read JSONL file."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_correct_flag(v: Any) -> bool:
    """Check if answer is marked correct."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return False


def _extract_gold(answers: list[dict[str, Any]]) -> str:
    """Extract gold answer from answer list."""
    for a in answers or []:
        if _is_correct_flag(a.get("correct")):
            return str(a.get("text", ""))
    return ""


def _prepare_case(raw: dict[str, Any]) -> dict[str, Any]:
    """Prepare a case for the environment."""
    answers = raw.get("answers", []) or []
    return {
        "question": raw.get("question", ""),
        "patient_info": raw.get("patient_info", ""),
        "physical_exams": raw.get("physical_exams", ""),
        "answers": answers,
        "image_url": raw.get("image_url", "") or raw.get("image", ""),
        "gold": _extract_gold(answers),
        "type": raw.get("type", []),
    }


def _brace_content(text: str) -> str:
    """Extract content from {braces}."""
    matches = re.findall(r"\{([^}]+)\}", text)
    return matches[-1].strip() if matches else ""


def _normalize(s: str) -> str:
    """Normalize string for comparison."""
    if not s:
        return ""
    s = s.strip().lower()
    s = s.strip("{}(). ")
    return s


def _to_text(completion: Any) -> str:
    """Extract text from completion."""
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", ""))
    return str(completion or "")


def _last_assistant_text(messages: vf.Messages) -> str:
    """Get last assistant message text."""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return str(m.get("content", ""))
    return ""


def _log_debug(line: str) -> None:
    """Write debug log."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _build_prompt(case: dict[str, Any]) -> dict[str, str]:
    """Build initial prompt for a case."""
    opts = "\n".join(f"- {opt.get('text', '')}" for opt in case.get("answers", []))
    px = (
        case.get("physical_exams")
        or "No additional examination or test results were reported."
    )
    user = (
        "You are interacting with a standardized patient. Ask for HISTORY, EXAM, TESTS, or IMAGE "
        "as needed to reach a final diagnosis.\n\n"
        "Objective: Assess and diagnose the patient based on the provided information and medical image.\n\n"
        f"Case Information:\n{case.get('question', '')}\n\n"
        "Patient Information:\n"
        f"{case.get('patient_info', '')}\n\n"
        "Physical Examination and Tests:\n"
        f"{px}\n\n"
        "Available Information:\n"
        "- Patient history and symptoms\n"
        "- Physical examination findings and test results\n"
        "- Medical image (if applicable)\n\n"
        "Answer Choices (choose EXACTLY one):\n"
        f"{opts}\n\n"
        "Instructions:\n"
        "1) Ask for HISTORY, EXAM, TESTS, or IMAGE as needed (you can request them by name).\n"
        "2) When ready, provide ONE diagnosis as EXACTLY one of the answer choices inside {Diagnosis} "
        "with no extra words and no punctuation inside the braces."
    )
    return {"system": SYSTEM_PROMPT, "user": user}


# ---------- Reward Function ----------
def accuracy_reward(
    prompt: str,  # noqa: ARG001 - Required by verifiers interface
    completion: Any,
    info: dict[str, Any],
) -> float:
    """Compute accuracy reward for diagnosis."""
    gold = (info or {}).get("gold", "") or ""
    answers = (info or {}).get("answers", []) or []

    comp_text = _to_text(completion)
    pred = _brace_content(comp_text) or comp_text
    pred_clean = pred.strip().strip(".{} ")

    # If gold missing, fallback to correct answer from answers
    if not gold and answers:
        gold = _extract_gold(answers)

    gold_norm = _normalize(gold)
    pred_norm = _normalize(pred_clean)

    ok = gold_norm and (pred_norm == gold_norm)

    _log_debug(
        f"[score] raw_gold={gold!r} raw_pred={pred_clean!r} "
        f"norm_gold={gold_norm!r} norm_pred={pred_norm!r} ok={ok}"
    )

    # Extra fallback: check if pred_norm equals any other correct option
    if not ok and answers:
        for opt in answers:
            if _normalize(opt.get("text", "")) == pred_norm and _is_correct_flag(
                opt.get("correct")
            ):
                ok = True
                break

    return 1.0 if ok else 0.0


class AgentClinicAccuracyReward(BaseRewardFunction):
    """Verifiers-compatible reward function for AgentClinic NEJM.

    Wraps the accuracy_reward function to match the radiant_harness
    BaseRewardFunction interface.
    """

    def __call__(
        self,
        prompt: str,
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        """Compute accuracy reward for diagnosis."""
        return accuracy_reward(prompt, completion, info)


# ---------- Environment ----------
class AgentClinicNEJMMultiTurn(vf.MultiTurnEnv):
    """Multi-turn environment for AgentClinic NEJM diagnostic reasoning."""

    def __init__(
        self,
        cases: list[dict[str, Any]] | None = None,
        *,
        dataset_path: str | None = None,
        max_turns: int = 10,
        name: str = "AgentClinicNEJM",
    ) -> None:
        """Initialize environment.

        Args:
            cases: Pre-loaded cases (optional)
            dataset_path: Path to JSONL dataset file
            max_turns: Maximum conversation turns
            name: Environment name
        """
        if cases is None:
            data_path = dataset_path or DEFAULT_DATASET_PATH
            raw_cases = _read_jsonl(data_path)
        else:
            raw_cases = cases

        prepared = [_prepare_case(c) for c in raw_cases]

        prompts: list[list[dict[str, str]]] = []
        infos: list[dict[str, Any]] = []

        for idx, case in enumerate(prepared):
            p = _build_prompt(case)
            prompts.append([
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ])
            infos.append({
                "case_index": idx,
                "gold": case.get("gold", ""),
                "answers": case.get("answers", []),
                "image_url": case.get("image_url", ""),
                "question": case.get("question", ""),
                "patient_info": case.get("patient_info", ""),
                "physical_exams": case.get("physical_exams", ""),
            })

        dataset = Dataset.from_dict({
            "id": list(range(len(prepared))),
            "prompt": prompts,
            "info": infos,
        })

        super().__init__(name=name, dataset=dataset)
        self._cases = prepared
        self._max_turns = max_turns

        self.rubric = vf.Rubric(
            funcs=[accuracy_reward],
            weights=[1.0],
        )

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_initial_state(
        self,
        prompt: vf.Messages,  # noqa: ARG002 - Required by interface
        info: dict[str, Any],
    ) -> vf.State:
        """Build initial state for episode."""
        # Track whether the assistant has elicited at least one env reply
        return {"turn": 0, "info": info, "asked": False}

    async def is_completed(
        self,
        messages: vf.Messages,
        state: vf.State,
        info: dict[str, Any] | None = None,
    ) -> bool:
        """Check if episode is complete."""
        # Stop if we hit max_turns
        if state.get("turn", 0) >= self._max_turns:
            return True

        last_asst = _last_assistant_text(messages)
        if not last_asst:
            # Model hasn't replied yet → not complete
            return False

        # Require that the assistant has asked for at least one section
        if not state.get("asked", False):
            return False

        # Done if assistant provided brace-wrapped answer
        if _brace_content(last_asst):
            return True

        current_info = info or state.get("info", {}) or {}
        answers = current_info.get("answers", []) or []
        norm = _normalize(last_asst)
        return any(_normalize(str(opt.get("text", ""))) in norm for opt in answers)

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        info: dict[str, Any] | None = None,  # noqa: ARG002 - Required by interface
    ) -> tuple[vf.Messages, vf.State]:
        """Generate environment response to assistant message."""
        turn = state.get("turn", 0)
        new_state = dict(state)
        new_state["turn"] = turn + 1

        last_asst = _last_assistant_text(messages)

        # If assistant gave brace answer and has asked, end episode
        if last_asst and _brace_content(last_asst) and state.get("asked", False):
            return [], new_state

        # Reply with requested section
        text_lower = (last_asst or "").lower()
        case_info = state.get("info", {})

        asked = False  # Flip to True when delivering a requested section

        if "history" in text_lower or "symptom" in text_lower:
            reply = (
                f"Patient History\n"
                f"{case_info.get('patient_info', 'No additional history available.')}"
            )
            asked = True
        elif "exam" in text_lower or "physical" in text_lower:
            reply = (
                f"Physical Examination Findings\n"
                f"{case_info.get('physical_exams', 'No additional exam findings available.')}"
            )
            asked = True
        elif any(
            k in text_lower
            for k in ["test", "lab", "result", "imaging", "x-ray", "ct", "mri"]
        ):
            reply = (
                f"Test Results\n"
                f"{case_info.get('physical_exams', 'No additional test results available.')}"
            )
            asked = True
        elif "image" in text_lower or "photo" in text_lower or "picture" in text_lower:
            reply = (
                case_info.get("image_url")
                or "No medical image is available for this case."
            )
            asked = True
        else:
            reply = (
                "You can request HISTORY, EXAM, TESTS, or IMAGE. "
                "When ready, give ONE diagnosis inside {Diagnosis} with no extra words and no punctuation."
            )

        if asked:
            new_state["asked"] = True

        return [{"role": "user", "content": reply}], new_state


# ---------- Loader ----------
def load_environment(
    dataset_path: str | None = None,
    max_turns: int = 10,
    **kwargs: Any,
) -> vf.Environment:
    """Load AgentClinic NEJM environment.

    Args:
        dataset_path: Path to JSONL dataset
        max_turns: Maximum conversation turns
        **kwargs: Additional environment arguments

    Returns:
        Configured environment
    """
    return AgentClinicNEJMMultiTurn(
        cases=None,
        dataset_path=dataset_path or DEFAULT_DATASET_PATH,
        max_turns=max_turns,
        **kwargs,
    )
