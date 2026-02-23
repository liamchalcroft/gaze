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
    - JSON embedded in surrounding text

    Uses Python's JSONDecoder.raw_decode() for robust parsing that correctly
    handles JSON strings containing braces.

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

    # Try to parse directly first (most common case)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            typed_direct: dict[str, Any] = result
            return typed_direct
        return None
    except json.JSONDecodeError:
        pass

    # Use raw_decode to find JSON object - this correctly handles strings with braces
    decoder = json.JSONDecoder()
    # Find all potential JSON start positions
    for i, c in enumerate(text):
        if c != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text, i)
            if isinstance(result, dict):
                typed_decoded: dict[str, Any] = result
                return typed_decoded
        except json.JSONDecodeError:
            continue

    return None
