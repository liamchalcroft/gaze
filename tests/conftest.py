"""Shared pytest fixtures for the gaze test suite."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

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


# ---------------------------------------------------------------------------
# HTTP test doubles
# ---------------------------------------------------------------------------


@pytest.fixture
def make_mock_http_client() -> Any:
    """Fixture returning :func:`build_mock_http_client` (see below)."""
    return build_mock_http_client


@pytest.fixture
def json_route() -> Any:
    """Fixture returning :func:`build_json_route`."""
    return build_json_route


@pytest.fixture
def download_route() -> Any:
    """Fixture returning :func:`build_download_route`."""
    return build_download_route


def build_mock_http_client(handler: Any, **client_kwargs: Any) -> Any:
    """Build an ``httpx.AsyncClient`` whose requests are answered by *handler*.

    ``handler(request) -> httpx.Response`` runs in place of the network, so the
    real client code path is exercised (parameter encoding, ``raise_for_status``,
    JSON decoding) instead of a hand-built stand-in for it.
    """
    import httpx

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **client_kwargs)


def build_json_route(routes: dict[str, Any], default_status: int = 404) -> Any:
    """Return a MockTransport handler dispatching on a substring of the URL path.

    Each value is either a JSON-serializable body, or a ``(status, body)`` pair;
    a body given as ``bytes`` or ``str`` is returned verbatim as text.
    """
    import httpx

    def handler(request: Any) -> Any:
        for fragment, payload in routes.items():
            if fragment in str(request.url):
                status, body = payload if isinstance(payload, tuple) else (200, payload)
                if isinstance(body, bytes | str):
                    return httpx.Response(status, text=body, request=request)
                return httpx.Response(status, json=body, request=request)
        return httpx.Response(default_status, text="no route", request=request)

    return handler


def build_download_route(
    *,
    status: int = 200,
    content_type: str = "image/jpeg",
    content: bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100,
    content_length: str | None = None,
) -> Any:
    """Handler for image-download tests, including a lying Content-Length.

    ``content_length`` overrides the real header so the streaming size guard can
    be exercised against a server that misreports the body size.
    """
    import httpx

    def handler(request: Any) -> Any:
        headers = {"Content-Type": content_type}
        if content_length is not None:
            headers["Content-Length"] = content_length
        return httpx.Response(status, content=content, headers=headers, request=request)

    return handler
