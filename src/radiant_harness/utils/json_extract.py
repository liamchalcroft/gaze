"""Shared JSON extraction utilities for parsing model outputs."""

from __future__ import annotations

import json
from typing import Any

from beartype import beartype


@beartype
def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract JSON object from model output text.

    Handles common formats:
    - Markdown code blocks (```json ... ```)
    - Raw JSON objects
    - Nested JSON objects

    Args:
        text: Text that may contain a JSON object

    Returns:
        Parsed JSON dict, or None if no valid JSON found
    """
    text = text.strip()
    if not text:
        return None

    # Handle markdown code block
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline == -1:
            return None
        closing = text.rfind("```")
        if closing <= first_newline:
            return None
        text = text[first_newline + 1 : closing].strip()

    # Try to parse directly first
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return None
    except json.JSONDecodeError:
        pass

    # Find and extract JSON object with brace matching
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start : i + 1])
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass
                break

    return None
