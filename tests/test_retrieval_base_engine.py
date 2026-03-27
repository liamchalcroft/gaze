"""Tests for retrieval/base.py BaseSearchEngine session lifecycle."""

from __future__ import annotations

import aiohttp
import pytest

from radiant_harness.retrieval.base import BaseSearchEngine
from radiant_harness.retrieval.base import SearchEngineError


class _ConcreteEngine(BaseSearchEngine[dict, SearchEngineError]):
    """Minimal concrete engine for testing base class session management."""

    async def _search_impl(self, query: str, max_results: int) -> list[dict]:
        return [{"title": query, "n": max_results}]

    def _make_error(
        self, message: str, original_error: Exception | None = None
    ) -> SearchEngineError:
        return SearchEngineError(self.name, message, original_error)


class TestBaseSearchEngineSession:
    """Cover _get_session (lines 129-134) and close (lines 139-140)."""

    @pytest.mark.asyncio
    async def test_get_session_creates_client_session(self) -> None:
        engine = _ConcreteEngine("test")
        session = await engine._get_session()
        assert isinstance(session, aiohttp.ClientSession)
        assert not session.closed
        await engine.close()

    @pytest.mark.asyncio
    async def test_get_session_reuses_existing_session(self) -> None:
        engine = _ConcreteEngine("test")
        s1 = await engine._get_session()
        s2 = await engine._get_session()
        assert s1 is s2
        await engine.close()

    @pytest.mark.asyncio
    async def test_get_session_recreates_after_close(self) -> None:
        engine = _ConcreteEngine("test")
        s1 = await engine._get_session()
        await engine.close()
        s2 = await engine._get_session()
        assert s1 is not s2
        assert not s2.closed
        await engine.close()

    @pytest.mark.asyncio
    async def test_close_when_no_session_is_noop(self) -> None:
        engine = _ConcreteEngine("test")
        await engine.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_sets_session_to_none(self) -> None:
        engine = _ConcreteEngine("test")
        await engine._get_session()
        assert engine._session is not None
        await engine.close()
        assert engine._session is None

    @pytest.mark.asyncio
    async def test_config_property_returns_search_config(self) -> None:
        engine = _ConcreteEngine("test")
        assert engine.config.timeout_seconds > 0
        assert engine.config.max_retries >= 1
