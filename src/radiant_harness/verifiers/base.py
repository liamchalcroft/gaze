"""Base multi-turn environment for verifiers integration.

Provides a reusable template for implementing MultiTurnEnv with
Radiant Harness processors and tools.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset


class BaseMultiTurnEnv(vf.MultiTurnEnv):
    """Base multi-turn environment for Radiant Harness integration.

    Provides common functionality for multi-turn environments:
    - Dataset loading from JSONL files
    - Turn tracking and limits
    - Standard message processing
    - Logging utilities
    - Tool request parsing

    Subclasses should implement:
    - setup_state: Initialize episode state
    - env_response: Generate environment responses
    """

    def __init__(
        self,
        cases: list[dict[str, Any]] | None = None,
        *,
        dataset_path: str | None = None,
        max_turns: int = 10,
        name: str = "BaseRadiantEnv",
        log_dir: Path | str | None = None,
    ) -> None:
        """Initialize environment.

        Args:
            cases: Pre-loaded cases (optional)
            dataset_path: Path to JSONL dataset file
            max_turns: Maximum conversation turns
            name: Environment name
            log_dir: Directory for debug logs
        """
        self._max_turns = max_turns
        self._log_dir = Path(log_dir) if log_dir else Path(__file__).parent.parent / "logs"
        self._log_path = self._log_dir / f"{name.lower()}_debug.log"

        # Load cases
        if cases is None and dataset_path:
            cases = self._load_jsonl(dataset_path)
        elif cases is None:
            cases = []

        # Process cases into prompts and info
        prompts, infos = self._prepare_cases(cases)

        dataset = Dataset.from_dict(
            {
                "id": list(range(len(cases))),
                "prompt": prompts,
                "info": infos,
            }
        )

        super().__init__(max_turns=max_turns, dataset=dataset)
        self._cases = cases

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        """Load cases from JSONL file."""
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows

    def _prepare_cases(
        self,
        cases: list[dict[str, Any]],
    ) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
        """Process cases into prompts and info structures.

        Override this method to customize case processing.

        Args:
            cases: Raw case dictionaries

        Returns:
            Tuple of (prompts, infos) for dataset construction
        """
        prompts: list[list[dict[str, Any]]] = []
        infos: list[dict[str, Any]] = []

        for idx, case in enumerate(cases):
            # Default prompt structure
            prompt = self._build_prompt(case)
            prompts.append(prompt)

            # Default info structure
            info = {
                "case_index": idx,
                **case,  # Include all case data
            }
            infos.append(info)

        return prompts, infos

    def _build_prompt(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        """Build initial prompt for a case.

        Override this method to customize prompt construction.

        Args:
            case: Case dictionary

        Returns:
            List of message dicts
        """
        system_prompt = self.get_system_prompt()
        user_content = self._build_user_message(case)

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_message(self, case: dict[str, Any]) -> str | list[dict[str, Any]]:
        """Build user message content.

        Override this method to customize user messages.

        Args:
            case: Case dictionary

        Returns:
            String or list of content dicts (for multimodal)
        """
        # Default: just return the question or case text
        return case.get("question", str(case))

    def get_system_prompt(self) -> str:
        """Get system prompt for the environment.

        Override this method to provide custom system prompt.

        Returns:
            System prompt string
        """
        return "You are a helpful assistant. Respond accurately and concisely."

    async def setup_state(self, state: vf.State) -> vf.State:
        """Initialize episode state.

        Override this method to customize initial state.

        Args:
            state: State dict pre-populated by verifiers

        Returns:
            State with custom fields added
        """
        state["turn"] = 0
        state["tool_uses"] = 0
        return state

    @vf.stop
    async def _turn_limit_reached(self, state: vf.State) -> bool:
        """Stop when turn limit is reached."""
        return state.get("turn", 0) >= self._max_turns

    async def env_response(
        self,
        messages: vf.Messages,  # noqa: ARG002 - Required by interface
        state: vf.State,
        **kwargs: Any,  # noqa: ARG002 - Required by interface
    ) -> vf.Messages:
        """Generate environment response to assistant message.

        Override this method to implement custom environment behavior.
        Mutate state in-place to track turn progress.

        Args:
            messages: Conversation messages
            state: Current episode state (mutate in-place)
            **kwargs: Additional arguments from verifiers

        Returns:
            Response messages
        """
        # Default: increment turn counter, no response
        state["turn"] = state.get("turn", 0) + 1
        return []

    def _log_debug(self, line: str) -> None:
        """Write debug log entry."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _last_assistant_text(self, messages: vf.Messages) -> str:
        """Get text from last assistant message."""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "assistant":
                content = m.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle multimodal content
                    for item in content:
                        if item.get("type") == "text":
                            return item.get("text", "")
        return ""

    def _extract_tool_request(
        self,
        text: str,
        tools: list[str],
    ) -> tuple[str, list[Any]] | None:
        """Extract tool request from text.

        Override this method to support custom tool parsing.

        Args:
            text: Assistant message text
            tools: List of valid tool names

        Returns:
            Tuple of (tool_name, args) or None
        """
        text_upper = text.upper()

        for tool in tools:
            # Simple pattern: TOOL [args]
            pattern = f"{tool.upper()}\\s*\\[([^\\]]+)\\]"
            match = re.search(pattern, text_upper)
            if match:
                # Parse arguments (comma-separated)
                args_str = match.group(1).strip()
                try:
                    # Try parsing as numbers
                    args = [float(x.strip()) for x in args_str.split(",")]
                except ValueError:
                    # Keep as strings
                    args = [x.strip() for x in args_str.split(",")]
                return tool.lower(), args

        return None
