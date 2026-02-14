"""Experiment configuration for NOVA conference paper."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

MODELS: list[tuple[str, str]] = [
    ("google/gemini-3-flash-preview", "gemini-3-flash"),
    ("moonshotai/kimi-k2.5", "kimi-k2.5"),
    ("z-ai/glm-4.6v", "glm-4.6v"),
    ("qwen/qwen3-vl-235b-a22b-instruct", "qwen3-vl-235b"),
    ("qwen/qwen3-vl-30b-a3b-instruct", "qwen3-vl-30b"),
    ("qwen/qwen3-vl-235b-a22b-thinking", "qwen3-vl-235b-think"),
    # qwen3-vl-30b-thinking excluded: OpenRouter providers (Novita, SiliconFlow)
    # reject structured outputs for this model.
]

ABLATION_MODEL: str = "google/gemini-3-flash-preview"
ABLATION_LABEL: str = "gemini-3-flash"


def run_label(
    model_label: str,
    mode: str,
    use_tools: bool,
    use_web_search: bool,
    max_turns: int,
    task: str,
) -> str:
    """Build a slugified directory name for one run."""
    if use_tools and use_web_search:
        tools_slug = "tools-search"
    elif use_tools:
        tools_slug = "tools"
    elif use_web_search:
        tools_slug = "search"
    else:
        tools_slug = "notools"
    return f"{model_label}__{mode}__{tools_slug}__{max_turns}t__{task}"


def experiment_configs(
    experiment_name: str,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (run_label, kwargs_dict) for each run in the named experiment."""
    if experiment_name in ("main_results", "all"):
        yield from _main_results()
    if experiment_name in ("tool_ablation", "all"):
        yield from _tool_ablation()
    if experiment_name in ("turn_depth", "all"):
        yield from _turn_depth()
    if experiment_name in ("per_task", "all"):
        yield from _per_task()


def _main_results() -> Iterator[tuple[str, dict[str, Any]]]:
    for model_id, label in MODELS:
        for mode in ("single_turn", "agentic"):
            use_tools = mode == "agentic"
            slug = run_label(label, mode, use_tools, False, 10, "all")
            yield (
                slug,
                {
                    "model_name": model_id,
                    "mode": mode,
                    "use_tools": use_tools,
                    "use_web_search": False,
                    "max_turns": 10,
                    "task": "all",
                },
            )


def _tool_ablation() -> Iterator[tuple[str, dict[str, Any]]]:
    configs = [
        ("single_turn", False, False),
        ("agentic", False, False),
        ("agentic", True, False),
        ("agentic", False, True),
        ("agentic", True, True),
    ]
    for mode, tools, search in configs:
        slug = run_label(ABLATION_LABEL, mode, tools, search, 10, "all")
        yield (
            slug,
            {
                "model_name": ABLATION_MODEL,
                "mode": mode,
                "use_tools": tools,
                "use_web_search": search,
                "max_turns": 10,
                "task": "all",
            },
        )


def _turn_depth() -> Iterator[tuple[str, dict[str, Any]]]:
    for turns in (1, 2, 3, 5, 10, 15, 20, 30):
        slug = run_label(ABLATION_LABEL, "agentic", True, False, turns, "all")
        yield (
            slug,
            {
                "model_name": ABLATION_MODEL,
                "mode": "agentic",
                "use_tools": True,
                "use_web_search": False,
                "max_turns": turns,
                "task": "all",
            },
        )


def _per_task() -> Iterator[tuple[str, dict[str, Any]]]:
    for task in ("caption", "diagnosis", "localization", "all"):
        slug = run_label(ABLATION_LABEL, "agentic", True, False, 10, task)
        yield (
            slug,
            {
                "model_name": ABLATION_MODEL,
                "mode": "agentic",
                "use_tools": True,
                "use_web_search": False,
                "max_turns": 10,
                "task": task,
            },
        )
