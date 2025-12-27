"""NOVA Brain MRI environment for MedMarks evaluation.

This environment provides multi-turn VLM evaluation on brain MRI analysis tasks:
- Caption: Radiological description and findings
- Diagnosis: Primary and differential diagnoses
- Localization: Abnormality detection with bounding boxes

Compatible with:
- MedMarks leaderboard (medmarks.ai)
- Prime Intellect verifiers framework
- medarc-eval CLI tool

Usage:
    # Via medarc-eval CLI
    medarc-eval nova-brain-mri -m gpt-4o -n 100

    # Programmatically
    import verifiers as vf
    env = vf.load_environment("nova-brain-mri", split="test")
    results = env.evaluate(client, "gpt-4o", num_examples=100)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import verifiers as vf
from datasets import Dataset

# Type aliases
NOVATask = Literal["caption", "diagnosis", "localization", "all"]
Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class NOVAEnvConfig:
    """Configuration for NOVA Brain MRI environment."""

    task: NOVATask = "all"
    max_turns: int = 10
    use_tools: bool = True
    use_web_search: bool = False
    iou_threshold: float = 0.3
    data_dir: str | None = None


def load(
    split: Split = "test",
    task: NOVATask = "all",
    max_turns: int = 10,
    use_tools: bool = True,
    use_web_search: bool = False,
    iou_threshold: float = 0.3,
    data_dir: str | None = None,
    **kwargs: Any,
) -> NOVABrainMRIEnv:
    """Load NOVA Brain MRI environment.

    This is the main entry point for MedMarks/verifiers integration.

    Args:
        split: Dataset split to use ("train", "validation", "test")
        task: NOVA task type ("caption", "diagnosis", "localization", "all")
        max_turns: Maximum conversation turns per episode
        use_tools: Enable visual manipulation tools (zoom, crop, contrast, etc.)
        use_web_search: Enable PubMed/medical literature search
        iou_threshold: IoU threshold for localization reward
        data_dir: Path to NOVA dataset directory (optional, uses default if None)
        **kwargs: Additional arguments passed to environment

    Returns:
        Configured NOVABrainMRIEnv instance

    Example:
        >>> env = load(split="test", task="diagnosis", max_turns=5)
        >>> results = env.evaluate(client, "gpt-4o", num_examples=50)
    """
    config = NOVAEnvConfig(
        task=task,
        max_turns=max_turns,
        use_tools=use_tools,
        use_web_search=use_web_search,
        iou_threshold=iou_threshold,
        data_dir=data_dir,
    )
    return NOVABrainMRIEnv(split=split, config=config, **kwargs)


class NOVABrainMRIEnv(vf.MultiTurnEnv):
    """Multi-turn environment for NOVA brain MRI benchmark.

    Implements the verifiers MultiTurnEnv interface for MedMarks evaluation.
    Supports multi-task evaluation (caption, diagnosis, localization) with
    optional tool usage for image manipulation and literature search.

    Attributes:
        config: Environment configuration
        split: Dataset split being used
    """

    def __init__(
        self,
        split: Split = "test",
        config: NOVAEnvConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize NOVA environment.

        Args:
            split: Dataset split
            config: Environment configuration
            **kwargs: Additional arguments for MultiTurnEnv
        """
        self.config = config or NOVAEnvConfig()
        self.split = split
        self._processor = None  # Lazy initialization

        # Load dataset
        cases = self._load_cases(split)
        prompts, infos = self._prepare_cases(cases)

        dataset = Dataset.from_dict(
            {
                "id": list(range(len(cases))),
                "prompt": prompts,
                "info": infos,
            }
        )

        super().__init__(
            name="nova-brain-mri",
            dataset=dataset,
            **kwargs,
        )
        self._cases = cases

    def _load_cases(self, split: Split) -> list[dict[str, Any]]:
        """Load NOVA cases from dataset.

        Args:
            split: Dataset split to load

        Returns:
            List of case dictionaries
        """
        if self.config.data_dir:
            data_path = Path(self.config.data_dir) / f"{split}.jsonl"
        else:
            # Default path relative to package
            data_path = Path(__file__).parent.parent.parent.parent / "data" / f"nova_{split}.jsonl"

        if not data_path.exists():
            # Try alternative location in examples/nova/data
            alt_path = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "examples"
                / "nova"
                / "data"
                / f"{split}.jsonl"
            )
            if alt_path.exists():
                data_path = alt_path
            else:
                # Return empty for now - dataset will be provided externally
                return []

        cases = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        return cases

    def _prepare_cases(
        self,
        cases: list[dict[str, Any]],
    ) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
        """Prepare cases into prompts and info structures.

        Args:
            cases: Raw case dictionaries

        Returns:
            Tuple of (prompts, infos)
        """
        prompts = []
        infos = []

        for idx, case in enumerate(cases):
            prompt = self._build_prompt(case)
            prompts.append(prompt)

            info = {
                "case_index": idx,
                "task": self.config.task,
                **case,
            }
            infos.append(info)

        return prompts, infos

    def _build_prompt(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        """Build initial prompt for a case.

        Args:
            case: Case dictionary with image_path, clinical_history, etc.

        Returns:
            List of message dicts
        """
        system_prompt = self._get_system_prompt()

        # Build user message with image
        user_content = self._build_user_message(case)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        return messages

    def _get_system_prompt(self) -> str:
        """Get system prompt for brain MRI analysis."""
        tool_instructions = ""
        if self.config.use_tools:
            tool_instructions = """
You have access to visual manipulation tools:
- zoom(factor): Zoom in/out (0.5x to 4.0x)
- crop(x1, y1, x2, y2): Crop to region (normalized 0-1)
- adjust_contrast(factor): Adjust contrast (0.5x to 3.0x)
- threshold(lower, upper): Apply intensity thresholding
- reset(): Restore original image

Use these tools to examine regions of interest more closely."""

        search_instructions = ""
        if self.config.use_web_search:
            search_instructions = """
You can search medical literature:
- search_web(query): Search PubMed for relevant articles
- search_images(query, modality): Find similar reference images

Use these to support your diagnostic reasoning with evidence."""

        task_description = {
            "caption": "radiological description and findings",
            "diagnosis": "primary diagnosis with differentials",
            "localization": "abnormality detection with bounding boxes",
            "all": "complete analysis: caption, diagnosis, and localization",
        }[self.config.task]

        return f"""You are an expert neuroradiologist analyzing brain MRI images.

Your task is to provide {task_description}.

{tool_instructions}
{search_instructions}

Response Format:
You must respond with valid JSON containing:
- caption: Radiological description with sequence characteristics and orientation
- diagnosis: Primary diagnosis with confidence, evidence, and differential diagnoses
- localization: Abnormality locations with bounding boxes [x1, y1, x2, y2] in pixels
- continue: true if you need more analysis (use tools), false when complete
- reasoning: Your chain-of-thought analysis

Be thorough and systematic. Examine the entire image before making conclusions.
Use tools when additional detail would help your analysis."""

    def _build_user_message(
        self,
        case: dict[str, Any],
    ) -> str | list[dict[str, Any]]:
        """Build user message with image support.

        Args:
            case: Case dictionary

        Returns:
            String or list of content dicts for multimodal
        """
        image_path = case.get("image_path") or case.get("image")
        history = case.get("clinical_history", "")
        modality = case.get("modality", "MRI")

        text_parts = ["Analyze this brain MRI image comprehensively."]

        if history:
            text_parts.append(f"\n**Clinical History:** {history}")

        if modality:
            text_parts.append(f"\n**Modality:** {modality}")

        text_parts.append("\nProvide complete captioning, diagnosis, and localization analysis.")

        text_content = "".join(text_parts)

        if image_path:
            # Return multimodal content
            return [
                {"type": "text", "text": text_content},
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{image_path}"},
                },
            ]

        return text_content

    def build_initial_state(
        self,
        prompt: vf.Messages,
        info: dict[str, Any],
    ) -> vf.State:
        """Build initial state for episode.

        Args:
            prompt: Initial messages
            info: Case information

        Returns:
            Initial state dictionary
        """
        state = {
            "turn": 0,
            "info": info,
            "tool_uses": 0,
            "is_complete": False,
        }

        # Store image path for tool execution
        image_path = info.get("image_path") or info.get("image")
        if image_path:
            state["image_path"] = image_path

        return state

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        info: dict[str, Any] | None = None,
    ) -> tuple[vf.Messages, vf.State]:
        """Generate environment response to assistant message.

        For NOVA, this handles tool execution and state updates.

        Args:
            messages: Conversation messages
            state: Current episode state
            info: Additional information

        Returns:
            Tuple of (response_messages, new_state)
        """
        new_state = dict(state)
        new_state["turn"] = state.get("turn", 0) + 1

        # Extract last assistant message
        last_msg = messages[-1] if messages else None
        if not last_msg or last_msg.get("role") != "assistant":
            return [], new_state

        content = last_msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = next(
                (item.get("text", "") for item in content if item.get("type") == "text"),
                "",
            )
        else:
            text = str(content)

        # Check for completion signal in JSON response using robust parsing
        decoder = json.JSONDecoder()
        for i, c in enumerate(text):
            if c != "{":
                continue
            try:
                response, _ = decoder.raw_decode(text, i)
                if isinstance(response, dict) and not response.get("continue", True):
                    new_state["is_complete"] = True
                break
            except json.JSONDecodeError:
                continue

        # Check turn limit
        if new_state["turn"] >= self.config.max_turns:
            new_state["is_complete"] = True

        return [], new_state

    async def is_completed(
        self,
        messages: vf.Messages,
        state: vf.State,
        info: dict[str, Any] | None = None,
    ) -> bool:
        """Check if episode is complete.

        Args:
            messages: Conversation messages
            state: Current episode state
            info: Additional information

        Returns:
            True if episode should end
        """
        if state.get("is_complete", False):
            return True
        if state.get("turn", 0) >= self.config.max_turns:
            return True
        return False

    def get_rubric(self) -> vf.Rubric:
        """Get evaluation rubric for this environment.

        Returns:
            Configured rubric with NOVA reward functions
        """
        from .rewards import create_nova_rubric

        return create_nova_rubric(
            task=self.config.task,
            iou_threshold=self.config.iou_threshold,
        )


# Export public API
__all__ = [
    "NOVABrainMRIEnv",
    "NOVAEnvConfig",
    "NOVATask",
    "Split",
    "load",
]
