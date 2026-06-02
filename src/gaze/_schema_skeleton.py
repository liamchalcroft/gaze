"""Response-schema skeletons and truncation-salvage helpers.

Split out of base.py; _build_schema_skeleton and _try_wrap_inner_schema
are re-exported from gaze.base.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def _build_prop_skeleton(prop: dict[str, Any]) -> Any:
    """Build a placeholder value for a single schema property.

    Recursively expands nested objects and arrays so that the skeleton
    shows the full expected structure, not just ``{...}`` placeholders.
    """
    ptype = prop.get("type", "string")
    if ptype == "boolean":
        return "true/false"
    if ptype == "array":
        items = prop.get("items", {})
        if items.get("type") == "object":
            return [_build_object_skeleton(items)]
        return ["..."]
    if ptype == "object":
        return _build_object_skeleton(prop)
    if ptype in ("number", "integer"):
        return 0
    enum_vals = prop.get("enum")
    return "|".join(enum_vals) if enum_vals else "..."


def _build_object_skeleton(schema_obj: dict[str, Any]) -> dict[str, Any]:
    """Build a nested skeleton dict from an object schema."""
    props = schema_obj.get("properties", {})
    skeleton: dict[str, Any] = {}
    for key, prop in props.items():
        skeleton[key] = _build_prop_skeleton(prop)
    return skeleton


def _collect_field_hints(
    props: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    """Recursively collect ``"- key: description"`` lines including nested fields."""
    hints: list[str] = []
    for key, prop in props.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        desc = prop.get("description", "")
        if desc:
            hints.append(f"- {full_key}: {desc}")
        # Recurse into nested objects
        if prop.get("type") == "object":
            hints.extend(_collect_field_hints(prop.get("properties", {}), full_key))
        # Recurse into array item objects
        if prop.get("type") == "array":
            items = prop.get("items", {})
            if items.get("type") == "object":
                hints.extend(_collect_field_hints(items.get("properties", {}), f"{full_key}[]"))
    return hints


def _build_schema_skeleton(
    response_schema: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Build a JSON skeleton and field hints from a response schema.

    Built once per analysis and reused for single-turn prompt injection
    and force-finalize nudges, avoiding redundant schema traversal.

    The skeleton recursively expands nested objects so models see the
    full expected structure (e.g. ``caption.description``) rather than
    opaque ``{...}`` placeholders.

    Returns:
        (skeleton_dict, field_hints) where skeleton_dict maps field names
        to placeholder values and field_hints is a list of
        ``"- key: description"`` lines (including nested fields).
    """
    if response_schema is None:
        return {"continue": False}, []

    schema_obj = response_schema.get("json_schema", {}).get("schema", {})
    props = schema_obj.get("properties", {})
    skeleton: dict[str, Any] = {}

    for key, prop in props.items():
        skeleton[key] = _build_prop_skeleton(prop)

    skeleton["continue"] = False
    field_hints = _collect_field_hints(props)
    return skeleton, field_hints


def _try_wrap_inner_schema(
    salvaged: dict[str, Any],
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    """Attempt to wrap a salvaged inner-object into the top-level schema.

    When a model is truncated mid-generation, the salvaged JSON may contain
    the fields of a sub-schema (e.g. caption's inner keys) instead of the
    expected top-level structure.  This function checks each top-level
    object property in the schema — if the salvaged keys are a subset of
    that property's own keys, it wraps the salvaged dict under that key
    and fills the remaining top-level keys with empty defaults.
    """
    schema_obj = response_schema.get("json_schema", {}).get("schema", {})
    props = schema_obj.get("properties", {})
    salvaged_keys = set(salvaged.keys()) - {"continue"}

    for prop_name, prop_def in props.items():
        if prop_def.get("type") != "object":
            continue
        inner_keys = set(prop_def.get("properties", {}).keys())
        if not inner_keys:
            continue
        # If salvaged keys overlap significantly with this sub-schema
        if salvaged_keys & inner_keys and salvaged_keys <= inner_keys:
            wrapped: dict[str, Any] = {
                prop_name: {k: v for k, v in salvaged.items() if k != "continue"}
            }
            # Fill remaining top-level keys with empty defaults
            for other_name, other_def in props.items():
                if other_name in wrapped or other_name == "continue":
                    continue
                otype = other_def.get("type", "string")
                if otype == "object":
                    wrapped[other_name] = {}
                elif otype == "array":
                    wrapped[other_name] = []
                elif otype in ("number", "integer"):
                    wrapped[other_name] = 0
                elif otype == "boolean":
                    wrapped[other_name] = False
                else:
                    wrapped[other_name] = ""
            wrapped["continue"] = False
            defaulted_keys = sorted(set(wrapped.keys()) - {prop_name, "continue"})
            logger.warning(
                f"Wrapped salvaged inner-schema keys {sorted(salvaged_keys)} under '{prop_name}'. "
                f"Fields filled with empty defaults (may produce zero evaluation scores): "
                f"{defaulted_keys}"
            )
            return wrapped
    return salvaged
