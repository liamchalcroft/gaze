"""NOVA brain-MRI evaluation environment for the Prime Intellect hub.

A MedMarks-style evaluation environment, built on the GAZE framework
(``gaze-vlm``), that scores the policy model under test on the NOVA
brain-MRI benchmark. GAZE supplies the JSON schema, the system and user
prompts, the JSON/text extraction helpers, and the reward functions. The
environment does not run its own agent loop: the policy model produces the
NOVA JSON and the rubric scores it.

Tasks: caption (token F1), diagnosis (top-1 plus coverage), and
localization (IoU-based detection F1 at the NOVA ACC50 threshold).

Usage:
    import verifiers as vf

    env = vf.load_environment("nova-brain-mri", split="test", task="all")
    results = env.evaluate(client=openai_client, model="gpt-4o", num_examples=100)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import verifiers as vf
from datasets import Dataset

from . import prompts
from ._utils import extract_json_from_text
from .rewards import create_nova_rubric

NOVATask = Literal["caption", "diagnosis", "localization", "all"]
Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class NOVAEnvConfig:
    """Configuration for the NOVA brain-MRI environment."""

    task: NOVATask = "all"
    max_turns: int = 10
    iou_threshold: float = 0.5
    data_dir: str | None = None


class NOVABrainMRIEnv(vf.MultiTurnEnv):
    """Multi-turn evaluation environment for the NOVA brain-MRI benchmark.

    Implements the verifiers ``MultiTurnEnv`` interface. The rubric scores the
    final NOVA JSON the policy model emits; no tools are executed.
    """

    def __init__(
        self,
        split: Split = "test",
        config: NOVAEnvConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or NOVAEnvConfig()
        self.split = split

        cases = self._load_cases(split)
        ids, prompt_msgs, infos = self._prepare_cases(cases)

        dataset = Dataset.from_dict(
            {
                "id": ids,
                "prompt": prompt_msgs,
                "info": infos,
            }
        )

        rubric = create_nova_rubric(
            task=self.config.task,
            iou_threshold=self.config.iou_threshold,
        )

        super().__init__(
            dataset=dataset,
            rubric=rubric,
            max_turns=self.config.max_turns,
            **kwargs,
        )

    def _load_cases(self, split: Split) -> list[dict[str, Any]]:
        """Load NOVA cases from the prepared JSONL for the split."""
        if self.config.data_dir:
            data_path = Path(self.config.data_dir) / f"{split}.jsonl"
        else:
            pkg_root = Path(__file__).parent.parent.parent  # environments/nova_brain_mri/
            data_path = pkg_root / "data" / f"nova_{split}.jsonl"

        if not data_path.exists():
            project_root = Path(__file__).parent.parent.parent.parent.parent
            alt_path = project_root / "examples" / "nova" / "data" / f"nova_{split}.jsonl"
            if alt_path.exists():
                data_path = alt_path
            else:
                msg = (
                    f"NOVA dataset not found. Searched:\n"
                    f"  - {data_path}\n"
                    f"  - {alt_path}\n"
                    f"Run prepare_data.py, or set NOVAEnvConfig.data_dir."
                )
                raise FileNotFoundError(msg)

        cases: list[dict[str, Any]] = []
        with open(data_path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line:
                    cases.append(json.loads(line))
        return cases

    def _prepare_cases(
        self,
        cases: list[dict[str, Any]],
    ) -> tuple[list[int], list[list[dict[str, Any]]], list[dict[str, Any]]]:
        """Build per-case ids, prompt messages, and info structures."""
        ids: list[int] = []
        prompt_msgs: list[list[dict[str, Any]]] = []
        infos: list[dict[str, Any]] = []

        for idx, case in enumerate(cases):
            ids.append(idx)
            prompt_msgs.append(self._build_prompt(case))
            infos.append({"case_index": idx, "task": self.config.task, **case})

        return ids, prompt_msgs, infos

    def _build_prompt(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the initial system and user messages for a case.

        Both messages use list-of-parts content so a HuggingFace ``Dataset``
        infers a single consistent type for the prompt column.
        """
        system_prompt = self._get_system_prompt()
        user_content = self._build_user_message(case)
        if isinstance(user_content, str):
            user_content = [{"type": "text", "text": user_content}]

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

    def _get_system_prompt(self) -> str:
        """Return the GAZE-style NOVA system prompt for the configured task."""
        return prompts.build_system_prompt(self.config.task)

    def _build_user_message(
        self,
        case: dict[str, Any],
    ) -> str | list[dict[str, Any]]:
        """Build the user message, attaching the image part when present."""
        text_content = prompts.build_user_message(case)

        image_path = case.get("image_path") or case.get("image")
        if image_path:
            return [
                {"type": "text", "text": text_content},
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
            ]

        return text_content

    async def setup_state(self, state: vf.State) -> vf.State:
        """Initialize per-episode state."""
        state["turn"] = 0
        state["is_complete"] = False
        return state

    @staticmethod
    def _last_assistant_text(messages: vf.Messages) -> str | None:
        """Return the text of the latest assistant message, if any."""
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, dict) or last.get("role") != "assistant":
            return None

        content = last.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(texts)
        return str(content)

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        """Inspect the latest assistant message and decide whether to continue.

        If the assistant emitted parseable JSON with a falsy ``continue``, the
        episode completes. Otherwise the turn counter advances; while under the
        turn limit a single short nudge asks for the final JSON, and at the
        limit the episode completes.
        """
        text = self._last_assistant_text(messages)
        if text is None:
            state["is_complete"] = True
            return []

        response = extract_json_from_text(text)
        if response is not None and not response.get("continue", True):
            state["is_complete"] = True
            return []

        state["turn"] = state.get("turn", 0) + 1
        if state["turn"] >= self.config.max_turns:
            state["is_complete"] = True
            return []

        return [
            {
                "role": "user",
                "content": (
                    "Please finish your analysis and output the final JSON object "
                    'with "continue": false.'
                ),
            }
        ]

    async def is_completed(self, state: vf.State, **kwargs: Any) -> bool:
        """Return whether the episode should end."""
        return state.get("is_complete", False) or state.get("turn", 0) >= self.config.max_turns


def load_environment(
    split: Split = "test",
    task: NOVATask = "all",
    max_turns: int = 10,
    iou_threshold: float = 0.5,
    data_dir: str | None = None,
    **kwargs: Any,
) -> NOVABrainMRIEnv:
    """Hub entry point: construct the NOVA brain-MRI evaluation environment.

    Args:
        split: Dataset split to load.
        task: ``caption``, ``diagnosis``, ``localization``, or ``all``.
        max_turns: Maximum turns per episode.
        iou_threshold: IoU threshold for localization (NOVA ACC50).
        data_dir: Directory holding ``<split>.jsonl`` (optional).
        **kwargs: Forwarded to ``NOVABrainMRIEnv``.

    Returns:
        Configured ``NOVABrainMRIEnv``.
    """
    config = NOVAEnvConfig(
        task=task,
        max_turns=max_turns,
        iou_threshold=iou_threshold,
        data_dir=data_dir,
    )
    return NOVABrainMRIEnv(split=split, config=config, **kwargs)


# Backward-compatible alias for the older medarc-eval loader convention.
load = load_environment


__all__ = [
    "NOVABrainMRIEnv",
    "NOVAEnvConfig",
    "NOVATask",
    "Split",
    "load",
    "load_environment",
]
