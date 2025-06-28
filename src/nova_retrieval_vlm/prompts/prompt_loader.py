"""
Prompt loading utilities for the NOVA medical image analysis system.

This module provides functions to load and combine system prompts with task-specific
Jinja templates, creating comprehensive prompts for different analysis modes.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger

from .system_prompts import get_system_prompt


def load_prompt(
    template_name: str,
    image_path: Path,
    passages: list[str],
    metadata: Dict[str, Any],
    mode: str = "baseline",
    system_prompt_override: Optional[str] = None,
) -> str:
    """
    Load and render a prompt template with system prompt integration.
    
    Args:
        template_name: Name of the Jinja template file (e.g., 'baseline/diagnosis.jinja')
        image_path: Path to the image file
        passages: List of retrieved passages (if using retrieval)
        metadata: Dictionary of metadata for the prompt
        mode: Analysis mode ('baseline', 'multiturn', 'visual', 'retrieval', 'web_search', 'comprehensive')
        system_prompt_override: Optional custom system prompt to override the default
        
    Returns:
        Rendered prompt string
    """
    # Get the base system prompt for the mode
    base_system_prompt = get_system_prompt(mode)
    
    # Use override if provided, otherwise use the base system prompt
    system_prompt = system_prompt_override or base_system_prompt
    
    # Load the task-specific template
    task_prompt = load_jinja_template(template_name, image_path, passages, metadata)
    
    # Combine system prompt with task-specific prompt
    combined_prompt = combine_prompts(system_prompt, task_prompt, mode)
    
    return combined_prompt


def load_jinja_template(
    template_name: str,
    image_path: Path,
    passages: list[str],
    metadata: Dict[str, Any],
) -> str:
    """
    Load and render a Jinja template with the given context.
    
    Args:
        template_name: Name of the Jinja template file
        image_path: Path to the image file
        passages: List of retrieved passages
        metadata: Dictionary of metadata
        
    Returns:
        Rendered template string
    """
    # Get the prompts directory
    prompts_dir = Path(__file__).parent
    env = Environment(loader=FileSystemLoader(str(prompts_dir)))
    
    try:
        template = env.get_template(template_name)
        context = {
            "image_path": str(image_path),
            "passages": passages,
            **metadata,
        }
        return template.render(**context)
    except Exception as e:
        logger.error(f"Failed to load template {template_name}: {e}")
        # Fallback to a basic prompt
        return f"Please analyze the image at {image_path} with the following context: {metadata}"


def combine_prompts(system_prompt: str, task_prompt: str, mode: str) -> str:
    """
    Combine system prompt with task-specific prompt based on mode.
    
    Args:
        system_prompt: The base system prompt for the mode
        task_prompt: The task-specific prompt from Jinja template
        mode: Analysis mode
        
    Returns:
        Combined prompt string
    """
    if mode == "baseline":
        # For baseline mode, use a simple combination
        return f"{system_prompt}\n\n{task_prompt}"
    
    elif mode == "multiturn":
        # For multiturn mode, emphasize the iterative reasoning process
        return f"{system_prompt}\n\n<task_instructions>\n{task_prompt}\n</task_instructions>\n\nBegin your multi-turn analysis process."
    
    elif mode == "visual":
        # For visual mode, emphasize visual operations and web search capabilities
        return f"{system_prompt}\n\n<visual_analysis_instructions>\n{task_prompt}\n</visual_analysis_instructions>\n\nBegin your visual analysis with appropriate operations."
    
    elif mode == "retrieval":
        # For retrieval mode, emphasize evidence-based analysis
        return f"{system_prompt}\n\n<retrieval_analysis_instructions>\n{task_prompt}\n</retrieval_analysis_instructions>\n\nBegin your retrieval-augmented analysis."
    
    elif mode == "web_search":
        # For web search mode, emphasize real-time information gathering
        return f"{system_prompt}\n\n<web_search_instructions>\n{task_prompt}\n</web_search_instructions>\n\nBegin your web search-augmented analysis."
    
    elif mode == "comprehensive":
        # For comprehensive mode, emphasize all capabilities
        return f"{system_prompt}\n\n<comprehensive_analysis_instructions>\n{task_prompt}\n</comprehensive_analysis_instructions>\n\nBegin your comprehensive analysis using all available capabilities."
    
    else:
        # Default fallback
        return f"{system_prompt}\n\n{task_prompt}"


def get_mode_from_template(template_name: str) -> str:
    """
    Infer the analysis mode from the template name.
    
    Args:
        template_name: Name of the Jinja template
        
    Returns:
        Inferred mode string
    """
    if template_name.startswith("multiturn/"):
        return "multiturn"
    elif template_name.startswith("visual_multiturn/"):
        return "visual"
    elif template_name.startswith("retrieval_"):
        return "retrieval"
    elif template_name.startswith("baseline/"):
        return "baseline"
    else:
        return "baseline"


def create_enhanced_prompt(
    template_name: str,
    image_path: Path,
    passages: list[str],
    metadata: Dict[str, Any],
    mode: Optional[str] = None,
    system_prompt_override: Optional[str] = None,
) -> str:
    """
    Create an enhanced prompt with automatic mode detection and system prompt integration.
    
    Args:
        template_name: Name of the Jinja template file
        image_path: Path to the image file
        passages: List of retrieved passages
        metadata: Dictionary of metadata
        mode: Optional explicit mode override
        system_prompt_override: Optional custom system prompt
        
    Returns:
        Enhanced prompt string
    """
    # Auto-detect mode if not provided
    if mode is None:
        mode = get_mode_from_template(template_name)
    
    # Load the enhanced prompt
    return load_prompt(
        template_name=template_name,
        image_path=image_path,
        passages=passages,
        metadata=metadata,
        mode=mode,
        system_prompt_override=system_prompt_override,
    )


# Backward compatibility - keep the original function signature
def load_prompt_legacy(
    template_name: str,
    image_path: Path,
    passages: list[str],
    metadata: Dict[str, Any],
) -> str:
    """
    Legacy function for backward compatibility.
    Uses the original Jinja-only approach without system prompt integration.
    """
    return load_jinja_template(template_name, image_path, passages, metadata) 