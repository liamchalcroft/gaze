"""
GEMeX-ThinkVG multi-turn evaluation environment

- Assistant analyzes chest X-ray images with visual grounding
- Can request ZOOM, CROP, CONTRAST, THRESHOLD, SEARCH operations
- Episode completes when assistant provides structured response with answer + bbox
- Reward = weighted combination of answer accuracy, location match, and IoU
"""

from __future__ import annotations

import json
import os
from typing import Any
from typing import List

import verifiers as vf
from datasets import Dataset

from radiant_harness.utils import extract_json_from_text

from ..rewards import GEMeXVerifiersReward
from ..rewards import RewardWeights
from ..schemas import parse_thinkvg_response
from ..schemas import validate_gemex_response

# ---------- Config ----------
SYSTEM_PROMPT = """You are an expert radiologist analyzing chest X-ray images.
Your task is to answer medical questions with visual grounding.

## Response Format

You must provide a structured response with:
1. **Reasoning**: Chain-of-thought analysis examining relevant anatomical regions
2. **Answer**: Clear, concise answer to the question
3. **Location**: Visual grounding with:
   - `reference`: Anatomical region name (e.g., 'right lower lobe', 'bilateral lung')
   - `bbox`: Bounding box [x1, y1, x2, y2] in pixel coordinates [0-336]

## Available Tools

You can request additional analysis by asking for:
- ZOOM [x1,y1,x2,y2]: Magnify a specific region
- CROP [x1,y1,x2,y2]: Focus on specific area
- CONTRAST [factor]: Adjust contrast (0.5-2.0)
- THRESHOLD [low,high]: Highlight intensity range
- SEARCH [query]: Find medical literature

## Final Response Format

When ready, provide your final answer as JSON:
```json
{
  "reasoning": "Your chain-of-thought analysis...",
  "answer": "Your diagnosis/finding",
  "location": {
    "reference": "anatomical region",
    "bbox": [x1, y1, x2, y2]
  }
}
```

Image dimensions: 336 x 336 pixels."""

LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
LOG_PATH = os.path.join(LOG_DIR, "debug.log")


# ---------- Tools ----------
def zoom_tool(x1: int, y1: int, x2: int, y2: int) -> str:
    """Zoom into a rectangular region. Coordinates are pixel indices [0, 336]."""
    coords = [x1, y1, x2, y2]
    return f"[ZOOM applied to region {coords}] Zoomed region shows enhanced detail. Continue analysis."


def crop_tool(x1: int, y1: int, x2: int, y2: int) -> str:
    """Crop the image to the specified region. Coordinates are pixel indices [0, 336]."""
    coords = [x1, y1, x2, y2]
    return f"[CROP applied to region {coords}] Cropped region ready. Continue analysis."


def contrast_tool(factor: float = 1.5) -> str:
    """Adjust image contrast by the given factor (0.5-2.0 recommended)."""
    return f"[CONTRAST adjusted by factor {factor}] Image contrast enhanced. Continue analysis."


def threshold_tool(low: int, high: int) -> str:
    """Apply intensity thresholding with low/high bounds (0-255)."""
    return f"[THRESHOLD applied: {low}-{high}] Thresholding applied. Continue analysis."


def search_tool(query: str) -> str:
    """Search medical literature for the given query."""
    return f"[SEARCH results for '{query}'] Medical literature search completed. Continue analysis."


# ---------- Helpers ----------
def _log_debug(line: str) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _last_assistant_text(messages: vf.Messages) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
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
            return str(content)
    return ""


def _extract_json_response(text: str) -> dict[str, Any] | None:
    """Extract JSON response from assistant message.

    Uses the shared ``extract_json_from_text`` utility (raw_decode based)
    so that this parser agrees with the reward function's parser.
    Falls back to XML-style ``parse_thinkvg_response`` with defaults
    injected for ``reasoning``/``confidence``/``continue``.
    """
    result = extract_json_from_text(text)
    if result is not None:
        return result

    xml_result = parse_thinkvg_response(text)
    if xml_result is not None:
        xml_result.setdefault("reasoning", "")
        xml_result.setdefault("confidence", 0.5)
        xml_result.setdefault("continue", False)
        return xml_result

    return None


# Normalise raw question-type strings from the HuggingFace dataset
# (e.g. "closed_ended_questions" → "closed_ended") so that downstream
# reward weighting always matches.
_QUESTION_TYPE_MAP: dict[str, str] = {
    "open_ended_questions": "open_ended",
    "closed_ended_questions": "closed_ended",
    "single_choice_questions": "single_choice",
    "multi_choice_questions": "multi_choice",
}


def _prepare_case(raw: dict[str, Any]) -> dict[str, Any]:
    """Prepare a GEMeX case for the environment."""
    raw_qt = raw.get("question_type", "open_ended")
    question_type = _QUESTION_TYPE_MAP.get(raw_qt, raw_qt)
    return {
        "question": raw.get("question", ""),
        "question_type": question_type,
        "image_path": raw.get("image_path", ""),
        "image_url": raw.get("image_url", ""),
        "gold_answer": raw.get("answer", ""),
        "gold_location": raw.get("location_reference", raw.get("location_ref", "")),
        "gold_bbox": raw.get("bbox", [0, 0, 0, 0]),
        "options": raw.get("options", []),
    }


def _build_prompt(case: dict[str, Any]) -> dict[str, str]:
    """Build initial prompt for a case."""
    question = case.get("question", "")
    question_type = case.get("question_type", "open_ended")
    options = case.get("options", [])

    user_parts = [
        "Analyze this chest X-ray image.",
        "",
        f"**Question ({question_type}):** {question}",
    ]

    if options:
        user_parts.append("")
        user_parts.append("**Options:**")
        for i, opt in enumerate(options):
            user_parts.append(f"  {chr(65 + i)}. {opt}")

    user_parts.extend([
        "",
        "You may request ZOOM, CROP, CONTRAST, THRESHOLD, or SEARCH operations via tool calls (do not emit bracketed text as a substitute).",
        "When ready, provide your reasoning, answer, and visual grounding location as JSON.",
    ])

    return {
        "system": [{"type": "text", "text": SYSTEM_PROMPT}],
        "user": [{"type": "text", "text": "\n".join(user_parts)}],
    }


# ---------- Reward Function ----------
def _make_gemex_reward(weights: RewardWeights | None = None) -> Any:
    """Create a reward closure that delegates to GEMeXVerifiersReward.

    This ensures a single canonical code path for extraction, validation,
    and scoring — shared with the processor's reward function.
    """
    verifiers_reward = GEMeXVerifiersReward(weights=weights)

    def gemex_reward(prompt: str, completion: Any, info: dict[str, Any]) -> float:
        """Compute combined GEMeX reward."""
        reward = verifiers_reward(prompt, completion, info)
        _log_debug(
            f"[reward] gold_answer={info.get('gold_answer', '')[:50]} "
            f"reward={reward:.3f}"
        )
        return reward

    return gemex_reward


# ---------- Environment ----------
class GEMeXThinkVGToolEnv(vf.ToolEnv):
    """Multi-turn ToolEnv for GEMeX-ThinkVG visual grounding."""

    def __init__(
        self,
        cases: list[dict[str, Any]] | None = None,
        *,
        dataset_path: str | None = None,
        max_turns: int = 8,
        name: str = "GEMeXThinkVG",
        reward_weights: RewardWeights | None = None,
    ) -> None:
        """Initialize environment.

        Args:
            cases: Pre-loaded cases (optional)
            dataset_path: Path to JSONL dataset file
            max_turns: Maximum conversation turns
            name: Environment name
            reward_weights: Custom reward weights
        """
        if cases is None and dataset_path:
            cases = self._load_jsonl(dataset_path)
        elif cases is None:
            cases = []

        prepared = [_prepare_case(c) for c in cases]

        prompts: list[list[dict[str, str]]] = []
        infos: list[dict[str, Any]] = []

        for idx, case in enumerate(prepared):
            p = _build_prompt(case)

            # Build message list with image if available
            messages = [{"role": "system", "content": p["system"]}]

            # Add image to user message if available
            user_content: list[dict[str, Any]] = []
            if case.get("image_url"):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": case["image_url"]},
                })
            user_content.extend(p["user"])

            messages.append({"role": "user", "content": user_content})
            prompts.append(messages)

            infos.append({
                "case_index": idx,
                "gold_answer": case.get("gold_answer", ""),
                "gold_location": case.get("gold_location", ""),
                "gold_bbox": case.get("gold_bbox", [0, 0, 0, 0]),
                "question_type": case.get("question_type", "open_ended"),
                "image_path": case.get("image_path", ""),
                "image_url": case.get("image_url", ""),
            })

        dataset = Dataset.from_dict({
            "id": list(range(len(prepared))),
            "prompt": prompts,
            "info": infos,
        })

        # Set up rubric and tools — capture weights in reward closure
        self.rubric = vf.Rubric(
            funcs=[_make_gemex_reward(reward_weights)],
            weights=[1.0],
        )
        tools: List[Any] = [zoom_tool, crop_tool, contrast_tool, threshold_tool, search_tool]

        super().__init__(
            name=name,
            dataset=dataset,
            tools=tools,
            rubric=self.rubric,
            max_turns=max_turns,
        )
        self._cases = prepared
        self._max_turns = max_turns
        self._reward_weights = reward_weights

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        """Load cases from JSONL file."""
        cases: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        return cases

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_initial_state(
        self,
        prompt: vf.Messages,
        info: dict[str, Any],
    ) -> vf.State:
        """Build initial state for episode."""
        return {
            "turn": 0,
            "info": info,
            "tool_uses": 0,
            "has_zoomed": False,
            "has_cropped": False,
            "has_searched": False,
        }

    async def is_completed(
        self,
        messages: vf.Messages,
        state: vf.State,
        info: dict[str, Any] | None = None,
    ) -> bool:
        """Check if episode is complete."""
        if await super().is_completed(messages, state, info):
            return True

        # Stop at max turns
        if state.get("turn", 0) >= self._max_turns:
            return True

        last_asst = _last_assistant_text(messages)
        if not last_asst:
            return False

        # Check if assistant provided valid JSON response
        response = _extract_json_response(last_asst)
        if response and validate_gemex_response(response):
            # Respect explicit continue=true signal: the model wants another turn
            return response.get("continue") is not True

        return False

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        info: dict[str, Any] | None = None,
    ) -> tuple[vf.Messages, vf.State]:
        """Delegate tool handling to ToolEnv and track turns."""
        new_state = dict(state)
        new_state["turn"] = state.get("turn", 0) + 1
        response, updated_state = await super().env_response(messages, new_state, info)
        updated_state.setdefault("turn", new_state["turn"])
        return response, updated_state


# ---------- Loader ----------
def load_environment(
    dataset_path: str | None = None,
    cases: list[dict[str, Any]] | None = None,
    max_turns: int = 8,
    reward_weights: RewardWeights | None = None,
    **kwargs: Any,
) -> GEMeXThinkVGToolEnv:
    """Load GEMeX-ThinkVG environment.

    Args:
        dataset_path: Path to JSONL dataset
        cases: Pre-loaded cases
        max_turns: Maximum conversation turns
        reward_weights: Custom reward weights (default 0.4/0.3/0.3)
        **kwargs: Additional environment arguments

    Returns:
        Configured environment
    """
    return GEMeXThinkVGToolEnv(
        cases=cases,
        dataset_path=dataset_path,
        max_turns=max_turns,
        reward_weights=reward_weights,
        **kwargs,
    )
