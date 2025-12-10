#!/usr/bin/env python
"""NOVA example using verifiers integration utilities.

Shows how to use the new utilities for NOVA brain MRI analysis
with multi-turn RL training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from datasets import Dataset

# NOTE: Verifiers is a core dependency; ensure your environment is installed
# (e.g., `uv sync` or `pip install -e .`).
try:
    import verifiers as vf
except ImportError as exc:
    raise ImportError(
        "verifiers package is required. It ships as a core dependency; "
        "run `uv sync` or `pip install -e .` if your environment is missing it."
    ) from exc

from radiant_harness.verifiers import BaseMultiTurnEnv
from radiant_harness.verifiers import ExactMatchReward
from radiant_harness.verifiers import TokenF1Reward
from radiant_harness.verifiers import IoUReward
from radiant_harness.verifiers import CombinedReward


def _last_assistant_text(messages: vf.Messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


class NOVAToolEnv(vf.ToolEnv):
    """Multi-turn ToolEnv for NOVA brain MRI analysis."""

    def __init__(
        self,
        cases: list[dict[str, Any]] | None = None,
        *,
        dataset_path: str | None = None,
        max_turns: int = 6,
        name: str = "NOVA_MRI",
        enable_tools: bool = True,
    ) -> None:
        """Initialize NOVA environment.

        Args:
            cases: Pre-loaded cases (optional)
            dataset_path: Path to JSONL dataset file
            max_turns: Maximum conversation turns
            name: Environment name
            enable_tools: Whether to enable tool use
        """
        self.enable_tools = enable_tools
        self.available_tools = ["zoom", "crop", "contrast", "search"] if enable_tools else []

        if cases is None and dataset_path:
            with open(dataset_path, encoding="utf-8") as fh:
                cases = [json.loads(line) for line in fh if line.strip()]
        elif cases is None:
            cases = []

        prompts: list[list[dict[str, Any]]] = []
        infos: list[dict[str, Any]] = []

        for idx, case in enumerate(cases):
            prompt = [
                {"role": "system", "content": [{"type": "text", "text": self.get_system_prompt()}]},
                {"role": "user", "content": self._build_user_message(case)},
            ]
            prompts.append(prompt)
            infos.append({
                "case_index": idx,
                "findings": case.get("findings", ""),
                "location": case.get("location", ""),
                "bbox": case.get("bbox", []),
                "severity": case.get("severity", ""),
                "image_path": case.get("image_path", ""),
            })

        dataset = Dataset.from_dict({
            "id": list(range(len(cases))),
            "prompt": prompts,
            "info": infos,
        })

        tools: List[Any] = [nova_zoom, nova_crop, nova_contrast, nova_search] if enable_tools else []

        self.rubric = create_nova_reward_functions().get_rubric()

        super().__init__(
            name=name,
            dataset=dataset,
            tools=tools,
            rubric=self.rubric,
            max_turns=max_turns,
        )
        self._max_turns = max_turns

    def get_system_prompt(self) -> str:
        """Get NOVA system prompt."""
        prompt = """You are a neurology expert analyzing brain MRI scans.

Your task is to identify and localize abnormalities in the provided images.

Instructions:
1. Examine the MRI scan carefully
2. Use tools to investigate suspicious areas
3. Provide structured findings

"""
        if self.enable_tools:
            prompt += """Available tools:
- ZOOM [x1,y1,x2,y2]: Magnify a specific region
- CROP [x1,y1,x2,y2]: Focus on specific area
- CONTRAST [factor]: Adjust image contrast (0.5-2.0)
- SEARCH [query]: Find relevant medical literature

Use native tool calls to invoke these tools (do not emit bracketed text as a substitute).
"""
        prompt += "\nWhen ready, provide your analysis as JSON."

        return prompt

    def _build_user_message(self, case: dict[str, Any]) -> str | list[dict[str, Any]]:
        """Build user message with image."""
        content = []

        # Add image if available
        if case.get("image_path"):
            content.append({
                "type": "image_url",
                "image_url": {"url": case["image_path"]},
            })

        # Add question
        question = case.get("question", "Analyze the brain MRI for any abnormalities.")
        content.append({
            "type": "text",
            "text": f"{question}\n\nFindings: {case.get('findings', 'None specified')}",
        })

        return content

    def build_initial_state(
        self,
        prompt: vf.Messages,
        info: dict[str, Any],
    ) -> vf.State:
        """Build initial state for NOVA task."""
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
        """Check if NOVA analysis is complete."""
        if await super().is_completed(messages, state, info):
            return True

        # End at max turns
        if state.get("turn", 0) >= self._max_turns:
            return True

        last_asst = _last_assistant_text(messages)

        # Check for JSON response
        try:
            start = last_asst.find("{")
            if start != -1:
                depth = 0
                for i, c in enumerate(last_asst[start:], start):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            json.loads(last_asst[start : i + 1])
                            return True
        except json.JSONDecodeError:
            pass

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


def create_nova_reward_functions() -> CombinedReward:
    """Create reward functions for NOVA evaluation."""
    rewards = [
        ExactMatchReward(
            normalize=True,
            case_sensitive=False,
            strip_braces=True,
        ),
        TokenF1Reward(
            normalize=True,
            tokenize="word",
        ),
        IoUReward(
            iou_threshold=0.5,
            normalized=True,
        ),
    ]

    # Weights: findings matching (0.5), semantic similarity (0.3), localization (0.2)
    weights = [0.5, 0.3, 0.2]
    return CombinedReward(
        rewards=rewards,
        weights=weights,
    )


# ---------- Tools ----------
def nova_zoom(x1: int, y1: int, x2: int, y2: int) -> str:
    """Zoom into a rectangular region of the MRI."""
    coords = [x1, y1, x2, y2]
    return f"[ZOOM applied to region {coords}] Zoomed region ready. Continue analysis."


def nova_crop(x1: int, y1: int, x2: int, y2: int) -> str:
    """Crop the MRI to the specified region."""
    coords = [x1, y1, x2, y2]
    return f"[CROP applied to region {coords}] Cropped region ready. Continue analysis."


def nova_contrast(factor: float = 1.5) -> str:
    """Adjust image contrast by the given factor."""
    return f"[CONTRAST adjusted by factor {factor}] Contrast enhanced. Continue analysis."


def nova_search(query: str) -> str:
    """Search medical literature for the given query."""
    return f"[SEARCH results for '{query}'] Literature search completed. Continue analysis."


def main() -> None:
    """Demonstrate NOVA integration."""
    print("NOVA Multi-Turn Environment with Verifiers")
    print("=" * 50)

    # Create sample NOVA data
    nova_cases = [
        {
            "question": "Identify any abnormalities in this brain MRI",
            "findings": "Small infarct in left basal ganglia",
            "location": "left basal ganglia",
            "bbox": [120, 140, 160, 180],  # Normalized coordinates
            "severity": "mild",
            "image_path": "path/to/mri1.jpg",
        },
        {
            "question": "Is there evidence of hemorrhage?",
            "findings": "No acute hemorrhage detected",
            "location": "none",
            "bbox": [],
            "severity": "none",
            "image_path": "path/to/mri2.jpg",
        },
    ]

    # Create environment
    env = NOVAToolEnv(
        cases=nova_cases,
        max_turns=6,
        enable_tools=True,
        name="NOVA_MRI",
    )

    print(f"Created NOVA environment with {len(env.dataset)} cases")

    # Create reward functions
    reward_fn = create_nova_reward_functions()
    rubric = reward_fn.get_rubric()

    print(f"Created reward function with {len(reward_fn.rewards)} components")

    # Test a case
    print("\n--- Testing Case 0 ---")
    case_data = env.dataset[0]

    print("System prompt:")
    print(case_data["prompt"][0]["content"][:200] + "...")

    print("\nUser message:")
    if isinstance(case_data["prompt"][1]["content"], list):
        for item in case_data["prompt"][1]["content"]:
            if item.get("type") == "text":
                print(f"Text: {item.get('text', '')[:100]}...")
            else:
                print(f"Content: {item.get('type', '')}")
    else:
        print(case_data["prompt"][1]["content"][:100] + "...")

    print("\nInfo:")
    info = case_data["info"]
    print(f"Findings: {info.get('findings', '')}")
    print(f"Location: {info.get('location', '')}")

    # Show training setup
    print("\n--- Training Setup ---")
    print("```python")
    print("import verifiers as vf")
    print("")
    print("# Configure training")
    print("trainer = vf.RLTrainer(")
    print("    environment=env,")
    print("    model='gpt-4o',  # or your fine-tuned model")
    print("    reward_rubric=rubric,")
    print("    learning_rate=1e-5,")
    print("    batch_size=8,")
    print("    max_rollouts=4,")
    print("    epochs=5,")
    print(")")
    print("")
    print("# Train")
    print("trainer.train()")
    print("")
    print("# Evaluate")
    print("evaluator = vf.Evaluator(")
    print("    environment=env,")
    print("    model='your-checkpoint',")
    print("    reward_rubric=rubric,")
    print(")")
    print("results = evaluator.evaluate()")
    print("```")

    print("\n--- Key Features ---")
    print("✓ Multi-turn MRI analysis with tool use")
    print("✓ Structured JSON response format")
    print("✓ Combined reward functions")
    print("✓ Automatic tool request parsing")
    print("✓ Configurable turn limits")
    print("✓ Detailed logging and metrics")


if __name__ == "__main__":
    main()