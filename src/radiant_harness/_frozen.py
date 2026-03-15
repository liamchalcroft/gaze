"""Internal helpers for freezing and thawing JSON-like payloads."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any
from typing import overload


@overload
def deep_freeze(obj: Mapping[str, Any]) -> MappingProxyType[str, Any]: ...


@overload
def deep_freeze(obj: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]: ...


@overload
def deep_freeze(
    obj: str | int | float | bool | None,
) -> str | int | float | bool | None: ...


@overload
def deep_freeze(obj: Any) -> Any: ...  # noqa: ANN401


def deep_freeze(obj: Any) -> Any:  # noqa: ANN401
    """Recursively freeze a JSON-like structure."""
    if isinstance(obj, MappingProxyType):
        obj = dict(obj)
    if isinstance(obj, Mapping):
        return MappingProxyType({str(k): deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list | tuple):
        return tuple(deep_freeze(item) for item in obj)
    return obj


@overload
def deep_thaw(obj: Mapping[str, Any]) -> dict[str, Any]: ...


@overload
def deep_thaw(obj: list[Any] | tuple[Any, ...]) -> list[Any]: ...


@overload
def deep_thaw(
    obj: str | int | float | bool | None,
) -> str | int | float | bool | None: ...


@overload
def deep_thaw(obj: Any) -> Any: ...  # noqa: ANN401


def deep_thaw(obj: Any) -> Any:  # noqa: ANN401
    """Recursively convert frozen containers back to plain JSON-like objects."""
    if isinstance(obj, MappingProxyType):
        obj = dict(obj)
    if isinstance(obj, Mapping):
        return {str(k): deep_thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple | list):
        return [deep_thaw(item) for item in obj]
    return obj
