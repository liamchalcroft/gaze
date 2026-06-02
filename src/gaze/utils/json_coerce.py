# pyright: basic
"""Schema-driven type coercion for local model responses.

Local models (especially thinking models like Qwen 3.5) frequently return
strings where the schema expects numbers, booleans, or arrays.  This module
walks the JSON schema properties and coerces mismatched types in-place.
"""

from __future__ import annotations

from typing import Any

from beartype import beartype
from loguru import logger


def _coerce_value(value: Any, prop_schema: dict[str, Any]) -> Any:
    """Coerce a single value to match the expected schema type.

    Returns the original value unchanged when coercion is not applicable.
    """
    expected = prop_schema.get("type")

    if expected == "number" and isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value

    if expected == "integer" and isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return value

    if expected == "boolean" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
        return value

    # Local models frequently return 1/0 for boolean fields.
    if expected == "boolean" and isinstance(value, int) and not isinstance(value, bool):
        return bool(value)

    if expected == "array" and isinstance(value, str):
        return [value] if value else []

    if expected == "array" and isinstance(value, list):
        items_schema = prop_schema.get("items", {})
        items_type = items_schema.get("type")
        if items_type == "integer":
            coerced_ints: list[int] = []
            for v in value:
                if isinstance(v, str):
                    try:
                        coerced_ints.append(int(float(v)))
                    except ValueError:
                        return value
                elif isinstance(v, int | float):
                    coerced_ints.append(int(v))
                else:
                    return value
            return coerced_ints
        if items_type == "number":
            coerced_floats: list[float] = []
            for v in value:
                if isinstance(v, str):
                    try:
                        coerced_floats.append(float(v))
                    except ValueError:
                        return value
                elif isinstance(v, int | float):
                    coerced_floats.append(float(v))
                else:
                    return value
            return coerced_floats
        # Recurse into array items that are objects (e.g. localizations[].bounding_box)
        if items_type == "object":
            for item in value:
                if isinstance(item, dict):
                    _coerce_dict(item, items_schema, prefix="[]")

    return value


def _coerce_dict(
    data: dict[str, Any],
    schema: dict[str, Any],
    prefix: str = "",
) -> None:
    """Coerce values in *data* in-place according to *schema* properties.

    Recurses into nested objects and array items with object schemas to
    arbitrary depth (e.g. NOVA ``localizations[].bounding_box``).
    """
    properties = schema.get("properties", {})

    for key, prop_schema in properties.items():
        if key not in data:
            continue

        value = data[key]
        path = f"{prefix}.{key}" if prefix else key

        if prop_schema.get("type") == "object" and isinstance(value, dict):
            _coerce_dict(value, prop_schema, prefix=path)
            continue

        old = value
        new = _coerce_value(old, prop_schema)
        if new is not old:
            logger.debug(f"Coerced {path}: {type(old).__name__} -> {type(new).__name__}")
            data[key] = new


@beartype
def coerce_json_types(response: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Coerce response values to match JSON schema types, in place.

    Recurses into nested objects and array items to arbitrary depth. The
    ``response`` dict is mutated in place and also returned, so the call can be
    used fluently: ``parsed = coerce_json_types(parsed, schema)``.

    Args:
        response: Parsed JSON response dict (mutated in place).
        schema: The ``response_format`` dict, raw JSON Schema object, or
                the nested ``json_schema.schema`` sub-dict -- all accepted.

    Returns:
        The same ``response`` dict, after coercion.
    """
    props = schema
    if "json_schema" in props:
        props = props["json_schema"]
    if "schema" in props:
        props = props["schema"]

    _coerce_dict(response, props)
    return response
