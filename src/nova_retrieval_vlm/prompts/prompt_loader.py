from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import Sequence

TEMPLATES_DIR = Path(__file__).parent

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=False,
)

def load_prompt(
    template_name: str,
    img_path: Path,
    passages: Sequence[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    """
    Render a prompt template with image path, optional passages, and metadata (e.g., clinical_history).

    Args:
        template_name: Name of the template file (e.g., 'baseline.jinja').
        img_path: Path to the image to include.
        passages: List of retrieved passages.
        metadata: Dictionary of example metadata (e.g., clinical_history, step summaries, request count).

    Returns:
        Rendered prompt string.
    """
    template = env.get_template(template_name)
    md = metadata or {}
    return template.render(
        img_path=str(img_path),
        passages=passages or [],
        clinical_history=md.get("clinical_history", ""),
        image_id=md.get("image_id", ""),
        width=md.get("width", ""),
        height=md.get("height", ""),
        # Additional parameters for iterative retrieval
        step1_summary=md.get("step1_summary", ""),
        step2_summary=md.get("step2_summary", ""),
        clinical_history_integration=md.get("clinical_history_integration", ""),
        request_count=md.get("request_count", 0),
        additional_passages=md.get("additional_passages", []),
        retrieval_history=md.get("retrieval_history", []),
    ) 