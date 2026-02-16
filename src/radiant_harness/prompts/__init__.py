"""Prompt loading utilities for the radiology VLM agent harness.

Provides Jinja2 template loading and rendering for system and task prompts.
Dataset-specific implementations should provide their own prompts directory.

Templates use **strict undefined variable behavior**: any reference to a variable
not present in the context dict will raise ``TemplateError`` immediately, preventing
silent generation of incomplete prompts.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from beartype import beartype
from minijinja import Environment
from minijinja import TemplateError as MinijinjaTemplateError
from typing_extensions import assert_never

from radiant_harness.exceptions import TemplateError

# Module-level environment with strict undefined behavior.
# Re-used across all render calls to avoid per-call construction overhead.
_env = Environment(undefined_behavior="strict")


class AnalysisMode(str, Enum):
    """Analysis modes for prompt generation.

    Inherits from str for easy comparison with string values.
    """

    AGENTIC = "agentic"
    SINGLE_TURN = "single_turn"


@beartype
def load_template(
    template_path: Path,
    context: dict[str, Any],
) -> str:
    """Load and render a Jinja template with the given context.

    Uses strict undefined variable behavior: any variable referenced in the
    template but missing from *context* will raise ``TemplateError``.

    Args:
        template_path: Path to the Jinja template file
        context: Dictionary of variables for template rendering

    Returns:
        Rendered template string

    Raises:
        TemplateError: If template file doesn't exist, a required variable
            is missing, or rendering fails for any other reason.

    Example:
        from pathlib import Path
        from radiant_harness.prompts import load_template

        prompt = load_template(
            Path("prompts/system.jinja"),
            {"task": "diagnosis", "modality": "MRI"}
        )
    """
    try:
        with open(template_path, encoding="utf-8") as f:
            template_content = f.read()
    except FileNotFoundError as e:
        raise TemplateError(
            f"Template file not found: {template_path}",
            template_path=template_path,
            original_error=e,
        ) from e
    except OSError as e:
        raise TemplateError(
            f"Failed to read template file: {e}",
            template_path=template_path,
            original_error=e,
        ) from e

    try:
        return _env.render_str(template_content, **context)
    except (ValueError, TypeError, RuntimeError, MinijinjaTemplateError) as e:
        raise TemplateError(
            f"Failed to render template: {e}",
            template_path=template_path,
            original_error=e,
        ) from e


@beartype
def load_prompt(
    prompts_dir: Path,
    template_name: str,
    mode: str,
    context: dict[str, Any],
) -> str:
    """Load and render a prompt template from a prompts directory.

    Args:
        prompts_dir: Base directory containing mode subdirectories
        template_name: Name of the template file (e.g., 'task.jinja')
        mode: Analysis mode ('agentic' or 'single_turn')
        context: Dictionary of variables for template rendering

    Returns:
        Rendered prompt string

    Raises:
        ValueError: If mode directory doesn't exist
        TemplateError: If template file doesn't exist or rendering fails
    """
    valid_modes = [m.value for m in AnalysisMode]
    if mode not in valid_modes:
        raise ValueError(f"Unknown mode: {mode!r}. Valid modes: {valid_modes}")

    mode_dir = prompts_dir / mode
    if not mode_dir.exists():
        raise ValueError(f"Mode directory not found: {mode_dir}")

    template_path = mode_dir / template_name
    return load_template(template_path, context)


@beartype
def load_system_prompt(
    prompts_dir: Path,
    mode: str,
    context: dict[str, Any],
) -> str:
    """Load the system prompt for a given mode.

    Args:
        prompts_dir: Base directory containing mode subdirectories
        mode: Analysis mode ('agentic' or 'single_turn')
        context: Dictionary of variables for template rendering

    Returns:
        Rendered system prompt string
    """
    return load_prompt(prompts_dir, "system.jinja", mode, context)


@beartype
def load_task_prompt(
    prompts_dir: Path,
    mode: str,
    context: dict[str, Any],
    template_name: str = "task.jinja",
) -> str:
    """Load the task prompt for a given mode.

    Args:
        prompts_dir: Base directory containing mode subdirectories
        mode: Analysis mode ('agentic' or 'single_turn')
        context: Dictionary of variables for template rendering
        template_name: Name of the task template (default: 'task.jinja')

    Returns:
        Rendered task prompt string
    """
    return load_prompt(prompts_dir, template_name, mode, context)


@beartype
def combine_prompts(
    system_prompt: str,
    task_prompt: str,
    mode: str,
) -> str:
    """Combine system prompt with task-specific prompt.

    Args:
        system_prompt: The base system prompt
        task_prompt: The task-specific prompt
        mode: Analysis mode string (must be valid AnalysisMode value)

    Returns:
        Combined prompt string

    Raises:
        ValueError: If mode is not a valid AnalysisMode
    """
    try:
        mode_enum = AnalysisMode(mode)
    except ValueError:
        valid_modes = [m.value for m in AnalysisMode]
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid_modes}") from None

    if mode_enum == AnalysisMode.SINGLE_TURN:
        return (
            f"{system_prompt}\n\n"
            f"<analysis_instructions>\n{task_prompt}\n"
            f"</analysis_instructions>\n\n"
            f"Provide your complete analysis in this single response."
        )
    if mode_enum == AnalysisMode.AGENTIC:
        return (
            f"{system_prompt}\n\n"
            f"<agentic_analysis_instructions>\n{task_prompt}\n"
            f"</agentic_analysis_instructions>\n\n"
            f"Begin your agentic analysis process."
        )

    assert_never(mode_enum)


@beartype
def create_prompt(
    prompts_dir: Path,
    mode: str,
    context: dict[str, Any],
    template_name: str = "task.jinja",
) -> str:
    """Create a complete prompt by combining system and task prompts.

    Args:
        prompts_dir: Base directory containing mode subdirectories
        mode: Analysis mode ('agentic' or 'single_turn')
        context: Dictionary of variables for template rendering.
            Standard context variables for generic harness templates:
            - domain_expertise: Domain-specific expertise description
            - tool_documentation: Generated documentation for available tools
            - analysis_workflow: Domain-specific analysis workflow steps
            - task_instructions: The specific task to perform
            - image_info: Information about the image (dimensions, etc.)
            - context: Additional context (e.g., clinical history)
            - output_format: Description of expected output format
            - image_path: Path to the image being analyzed
        template_name: Name of the task template (default: 'task.jinja')

    Returns:
        Complete combined prompt string

    Example:
        from radiant_harness.prompts import create_prompt
        from radiant_harness.tools import ToolRegistry

        registry = ToolRegistry(tools=my_tools)
        prompt = create_prompt(
            prompts_dir=Path("my_prompts"),
            mode="agentic",
            context={
                "domain_expertise": "You are an expert radiologist...",
                "tool_documentation": registry.get_documenter().generate_prompt_documentation(),
                "task_instructions": "Analyze this brain MRI...",
                "image_path": "/path/to/image.png",
            },
        )
    """
    system_prompt = load_system_prompt(prompts_dir, mode, context)
    task_prompt = load_prompt(prompts_dir, template_name, mode, context)
    return combine_prompts(system_prompt, task_prompt, mode)
