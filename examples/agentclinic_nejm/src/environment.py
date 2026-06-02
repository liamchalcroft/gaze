"""AgentClinic NEJM multi-turn evaluation environment.

- Assistant asks; env replies with HISTORY / EXAM_AND_TESTS / IMAGE
- Episode completes when assistant provides diagnosis in {Diagnosis} braces
  AND has made at least one information request
- Reward = 0.8 * accuracy + 0.2 * token-F1 for partial credit

Provides verifiers-compatible environment with combined reward.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset
from PIL import Image

from gaze.tools import encode_image
from gaze.verifiers import BaseRewardFunction
from gaze.verifiers import CombinedReward
from gaze.verifiers import TokenF1Reward
from gaze.verifiers import extract_completion_text

# ---------- Config ----------
DEFAULT_DATASET_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "agentclinic_nejm_extended.jsonl"
)

SYSTEM_PROMPT = (
    "Think step by step. When you are ready, output ONE diagnosis inside {Diagnosis} "
    "with no extra words and no punctuation inside the braces."
)

LOG_DIR = Path(__file__).resolve().parent / "log"
LOG_PATH = LOG_DIR / "debug.log"


# ---------- Helpers ----------
def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read JSONL file."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped:
                rows.append(json.loads(stripped))
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


def _brace_content(text: str, answer_options: list[str] | None = None) -> str:
    """Extract content from {braces}.

    When answer_options is provided, returns the last match whose normalized
    form equals a known option.  Falls back to last match otherwise.
    """
    matches = re.findall(r"\{([^}]+)\}", text)
    if not matches:
        return ""
    if answer_options:
        norm_options = {_normalize(opt) for opt in answer_options}
        for match in reversed(matches):
            if _normalize(match) in norm_options:
                return match.strip()
        return ""
    return matches[-1].strip()


def _normalize(s: str) -> str:
    """Normalize string for comparison."""
    if not s:
        return ""
    s = s.strip().lower()
    s = s.strip("{}(). ")
    return s


def _last_assistant_text(messages: vf.Messages) -> str:
    """Get last assistant message text."""
    if isinstance(messages, str):
        return messages
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            if texts:
                return "\n".join(texts)
    return ""


def _log_debug(line: str) -> None:
    """Write debug log."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _message_image_url(image_url: str) -> str:
    """Convert local image files to data URLs for OpenAI-compatible servers."""
    if not image_url:
        return image_url
    if image_url.startswith(("http://", "https://", "data:")):
        return image_url

    path = Path(image_url)
    if not path.exists():
        return image_url

    with Image.open(path) as image:
        return encode_image(image).to_data_url()


def _build_prompt(case: dict[str, Any]) -> list[dict[str, str]]:
    """Build initial prompt messages for a case.

    Only includes the chief complaint and answer choices -- patient history
    and exam data must be gathered through multi-turn interaction.
    """
    opts = "\n".join(f"- {opt.get('text', '')}" for opt in case.get("answers", []))
    user = (
        "You are interacting with a standardized patient. Ask for HISTORY, EXAM_AND_TESTS, or IMAGE "
        "as needed to reach a final diagnosis.\n\n"
        "Objective: Assess and diagnose the patient based on gathered information.\n\n"
        f"Chief Complaint:\n{case.get('question', '')}\n\n"
        "Answer Choices (choose EXACTLY one):\n"
        f"{opts}\n\n"
        "Instructions:\n"
        "1) Ask for HISTORY, EXAM_AND_TESTS, or IMAGE as needed (you can request them by name).\n"
        "2) When ready, provide ONE diagnosis as EXACTLY one of the answer choices inside {Diagnosis} "
        "with no extra words and no punctuation inside the braces."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------- Reward Functions ----------
def accuracy_reward(
    prompt: Any,  # noqa: ARG001 - Required by verifiers interface
    completion: Any,
    info: dict[str, Any],
) -> float:
    """Compute accuracy reward for diagnosis."""
    gold = (info or {}).get("gold", "") or ""
    answers = (info or {}).get("answers", []) or []
    answer_texts = [str(opt.get("text", "")) for opt in answers]

    comp_text = extract_completion_text(completion)
    pred = _brace_content(comp_text, answer_texts) or comp_text
    pred_clean = pred.strip().strip(".{} ")

    if not gold and answers:
        gold = _extract_gold(answers)

    gold_norm = _normalize(gold)
    pred_norm = _normalize(pred_clean)

    ok = gold_norm and (pred_norm == gold_norm)

    _log_debug(
        f"[score] raw_gold={gold!r} raw_pred={pred_clean!r} "
        f"norm_gold={gold_norm!r} norm_pred={pred_norm!r} ok={ok}"
    )

    if not ok and answers:
        for opt in answers:
            if _normalize(opt.get("text", "")) == pred_norm and _is_correct_flag(
                opt.get("correct")
            ):
                ok = True
                break

    return 1.0 if ok else 0.0


class AgentClinicAccuracyReward(BaseRewardFunction):
    """Verifiers-compatible accuracy reward for AgentClinic NEJM."""

    def __call__(
        self,
        prompt: Any,  # noqa: ARG002 - Required by interface
        completion: Any,
        info: dict[str, Any],
    ) -> float:
        return accuracy_reward(prompt, completion, info)


_COMBINED_REWARD = CombinedReward(
    [AgentClinicAccuracyReward(), TokenF1Reward()],
    weights=[0.8, 0.2],
)


def combined_reward(
    prompt: Any,
    completion: Any,
    info: dict[str, Any],
) -> float:
    """Combined accuracy (0.8) + token-F1 (0.2) reward."""
    return _COMBINED_REWARD(prompt, completion, info)


# ---------- Environment ----------
class AgentClinicNEJMMultiTurn(vf.MultiTurnEnv):
    """Multi-turn environment for AgentClinic NEJM diagnostic reasoning."""

    def __init__(
        self,
        cases: list[dict[str, Any]] | None = None,
        *,
        dataset_path: str | None = None,
        max_turns: int = 10,
    ) -> None:
        if cases is None:
            data_path = dataset_path or DEFAULT_DATASET_PATH
            raw_cases = _read_jsonl(data_path)
        else:
            raw_cases = cases

        prepared = [_prepare_case(c) for c in raw_cases]

        prompts: list[list[dict[str, str]]] = []
        infos: list[dict[str, Any]] = []

        for idx, case in enumerate(prepared):
            prompts.append(_build_prompt(case))
            infos.append(
                {
                    "case_index": idx,
                    "gold": case.get("gold", ""),
                    "answers": case.get("answers", []),
                    "image_url": case.get("image_url", ""),
                    "question": case.get("question", ""),
                    "patient_info": case.get("patient_info", ""),
                    "physical_exams": case.get("physical_exams", ""),
                }
            )

        dataset = Dataset.from_dict(
            {
                "id": list(range(len(prepared))),
                "prompt": prompts,
                "info": infos,
            }
        )

        rubric = vf.Rubric(funcs=[combined_reward], weights=[1.0])

        super().__init__(
            max_turns=max_turns,
            dataset=dataset,
            rubric=rubric,
        )
        self._cases = prepared

    async def setup_state(self, state: vf.State) -> vf.State:
        """Initialize custom episode state fields."""
        state["asked"] = False
        return state

    @vf.stop
    async def diagnosis_given(self, state: vf.State) -> bool:
        """Stop when assistant has asked for info and provided a brace-wrapped diagnosis."""
        if not state.get("asked", False):
            return False
        trajectory = state.get("trajectory", [])
        if not trajectory:
            return False
        last_completion = trajectory[-1].get("completion", [])
        last_text = _last_assistant_text(last_completion)
        if not last_text:
            return False
        info = state.get("info", {}) or {}
        answers = info.get("answers", []) or []
        answer_texts = [str(opt.get("text", "")) for opt in answers]
        return bool(_brace_content(last_text, answer_texts))

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,  # noqa: ARG002 - Required by verifiers interface
    ) -> vf.Messages:
        """Generate environment response to assistant message."""
        last_asst = _last_assistant_text(messages)
        info = state.get("info", {}) or {}

        text_lower = (last_asst or "").lower()
        reply_content: str | list[dict[str, Any]] = ""

        if "history" in text_lower or "symptom" in text_lower:
            reply_content = (
                f"Patient History\n{info.get('patient_info', 'No additional history available.')}"
            )
            state["asked"] = True
        elif any(
            k in text_lower
            for k in [
                "exam",
                "physical",
                "test",
                "lab",
                "result",
                "imaging",
                "x-ray",
                "ct",
                "mri",
            ]
        ):
            reply_content = (
                "Physical Examination and Test Results\n"
                f"{info.get('physical_exams', 'No additional findings available.')}"
            )
            state["asked"] = True
        elif "image" in text_lower or "photo" in text_lower or "picture" in text_lower:
            image_url = info.get("image_url", "")
            if image_url:
                reply_content = [
                    {"type": "text", "text": "Medical Image:"},
                    {"type": "image_url", "image_url": {"url": _message_image_url(image_url)}},
                ]
            else:
                reply_content = "No medical image is available for this case."
            state["asked"] = True
        else:
            reply_content = (
                "You can request HISTORY, EXAM_AND_TESTS, or IMAGE. "
                "When ready, give ONE diagnosis inside {Diagnosis} "
                "with no extra words and no punctuation."
            )

        return [{"role": "user", "content": reply_content}]


# ---------- Loader ----------
def load_environment(
    dataset_path: str | None = None,
    max_turns: int = 10,
) -> AgentClinicNEJMMultiTurn:
    """Load AgentClinic NEJM environment."""
    return AgentClinicNEJMMultiTurn(
        dataset_path=dataset_path or DEFAULT_DATASET_PATH,
        max_turns=max_turns,
    )
