"""Shared JSON extraction utilities for parsing model outputs."""

from __future__ import annotations

import json
from typing import Any
from typing import cast

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
            return cast("dict[str, Any]", result)
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
                return cast("dict[str, Any]", result)
        except json.JSONDecodeError:
            continue

    # Last resort: try repairing truncated JSON by closing unclosed brackets.
    # Local models frequently get cut off mid-generation, producing parseable
    # fragments that just need closing delimiters.
    return _try_repair_truncated(text)


def _try_repair_truncated(text: str) -> dict[str, Any] | None:
    """Try to repair truncated JSON by closing unclosed brackets/braces.

    Tracks nesting depth (respecting strings and escapes) to determine
    which delimiters are missing, then tries several suffix strategies.
    """
    start = text.find("{")
    if start == -1:
        return None

    fragment = text[start:]
    if len(fragment) < 20:
        return None

    # Track nesting outside of string literals
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in fragment:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    if not stack:
        return None  # Already balanced — not a truncation issue

    closers = "".join("}" if o == "{" else "]" for o in reversed(stack))

    # Try repair strategies in order of likelihood:
    #   1. Just close brackets (truncated after a complete value)
    #   2. Close unclosed string + brackets (truncated mid-string value)
    #   3. Complete a truncated key as null + close (truncated mid-key)
    #   4. Null value + close (truncated after colon)
    #   5. Zero value + close (truncated mid-number)
    suffixes = [
        closers,
        '"' + closers,
        '": null' + closers,
        "null" + closers,
        "0" + closers,
    ]

    for suffix in suffixes:
        try:
            result = json.loads(fragment + suffix)
            if isinstance(result, dict):
                return cast("dict[str, Any]", result)
        except json.JSONDecodeError:
            continue

    return None
