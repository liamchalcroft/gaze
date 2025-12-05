"""Prompt loading utilities for the NOVA medical image analysis system.

This module provides functions to load and combine system prompts with task-specific
Jinja templates, creating comprehensive prompts for different analysis modes.
"""

from pathlib import Path
from typing import Any

from beartype import beartype
from minijinja import Environment


@beartype
def load_prompt(
    template_name: str,
    image_path: Path,
    passages: list[str],
    metadata: dict[str, Any],
    mode: str = "single_turn",
    system_prompt_override: str | None = None,
) -> str:
    """Load and render a prompt template with system prompt integration.

    Args:
        template_name: Name of the Jinja template file (relative to mode directory)
        image_path: Path to the image file
        passages: List of retrieved passages (if using retrieval)
        metadata: Dictionary of metadata for the prompt
        mode: Analysis mode ('single_turn' or 'agentic')
        system_prompt_override: Optional custom system prompt to override the default

    Returns:
        Rendered prompt string
    """
    # Get the base system prompt for the mode
    base_system_prompt = load_system_prompt(mode, metadata)

    # Use override if provided, otherwise use the base system prompt
    system_prompt = system_prompt_override or base_system_prompt

    # Load the task-specific template
    task_prompt = load_jinja_template(template_name, image_path, passages, metadata, mode)

    # Combine system prompt with task-specific prompt
    combined_prompt = combine_prompts(system_prompt, task_prompt, mode)

    return combined_prompt


@beartype
def load_system_prompt(mode: str, metadata: dict[str, Any]) -> str:
    """Load system prompt from Jinja template with dynamic tool flags.

    Args:
        mode: Analysis mode ('single_turn' or 'agentic')
        metadata: Dictionary of metadata including tool flags

    Returns:
        Rendered system prompt string
    """
    # Load system prompt template
    prompts_dir = Path(__file__).parent / mode
    system_template_path = prompts_dir / "system.jinja"

    with open(system_template_path, encoding="utf-8") as f:
        template_content = f.read()

    # Create minijinja environment and render
    env = Environment()

    context = metadata.copy()
    return env.render_str(template_content, **context)


@beartype
def load_jinja_template(
    template_name: str,
    image_path: Path,
    passages: list[str],
    metadata: dict[str, Any],
    mode: str,
) -> str:
    """Load and render a template with the given context using minijinja.

    Args:
        template_name: Name of the template file (relative to mode directory)
        image_path: Path to the image file
        passages: List of retrieved passages
        metadata: Dictionary of metadata
        mode: Analysis mode directory

    Returns:
        Rendered template string
    """
    # Load template file from mode-specific directory
    prompts_dir = Path(__file__).parent / mode
    template_path = prompts_dir / template_name

    with open(template_path, encoding="utf-8") as f:
        template_content = f.read()

    # Create minijinja environment and render
    env = Environment()

    context = {
        "image_path": str(image_path),
        "passages": passages,
        **metadata,
    }
    return env.render_str(template_content, **context)


@beartype
def combine_prompts(system_prompt: str, task_prompt: str, mode: str) -> str:
    """Combine system prompt with task-specific prompt based on mode.

    Args:
        system_prompt: The base system prompt for the mode
        task_prompt: The task-specific prompt from Jinja template
        mode: Analysis mode

    Returns:
        Combined prompt string
    """
    if mode == "single_turn":
        # For single-turn mode, emphasize comprehensive one-shot analysis
        return (
            f"{system_prompt}\n\n"
            f"<comprehensive_analysis_instructions>\n{task_prompt}\n"
            f"</comprehensive_analysis_instructions>\n\n"
            f"Provide your complete analysis in this single response."
        )

    elif mode == "agentic":
        # For agentic mode, emphasize tool usage and multi-turn reasoning
        return (
            f"{system_prompt}\n\n"
            f"<agentic_analysis_instructions>\n{task_prompt}\n"
            f"</agentic_analysis_instructions>\n\n"
            f"Begin your agentic analysis process."
        )

    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'agentic' or 'single_turn'.")


@beartype
def get_mode_from_template(template_name: str) -> str:
    """Infer the analysis mode from the template name.

    Args:
        template_name: Name of the Jinja template

    Returns:
        Inferred mode string
    """
    if template_name.startswith("agentic_"):
        return "agentic"
    elif template_name.startswith("single_turn_"):
        return "single_turn"
    else:
        return "single_turn"


@beartype
def create_enhanced_prompt(
    template_name: str,
    image_path: Path,
    passages: list[str],
    metadata: dict[str, Any],
    mode: str | None = None,
    system_prompt_override: str | None = None,
) -> str:
    """Create an enhanced prompt with automatic mode detection and system prompt integration.

    Args:
        template_name: Name of the Jinja template file (can include mode prefix)
        image_path: Path to the image file
        passages: List of retrieved passages
        metadata: Dictionary of metadata
        mode: Optional explicit mode override (if not in template_name)
        system_prompt_override: Optional custom system prompt

    Returns:
        Enhanced prompt string
    """
    # Auto-detect mode if not provided
    if mode is None:
        mode = get_mode_from_template(template_name)

    # Remove mode prefix from template name if present
    if template_name.startswith(f"{mode}_"):
        template_name = template_name[len(f"{mode}_") :]

    # Load the enhanced prompt
    return load_prompt(
        template_name=template_name,
        image_path=image_path,
        passages=passages,
        metadata=metadata,
        mode=mode,
        system_prompt_override=system_prompt_override,
    )
