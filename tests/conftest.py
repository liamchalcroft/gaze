"""Shared pytest fixtures for the gaze test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from gaze.config import reset_config


@pytest.fixture(autouse=True)
def _reset_global_config() -> Iterator[None]:
    """Reset global config to defaults after every test.

    Prevents config mutations in one test from leaking into another.
    Runs automatically for all tests via ``autouse=True``.
    """
    yield
    reset_config()
