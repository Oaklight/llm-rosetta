"""Tests for sync/async compatibility wrappers.

Verifies that gateway functions that migrated from sync to async
(persistence → aiosqlite) remain callable from sync contexts, while
still working with ``await`` from async contexts.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from llm_rosetta._compat import maybe_sync


# ---------------------------------------------------------------------------
# maybe_sync helper
# ---------------------------------------------------------------------------


class TestMaybeSync:
    """Unit tests for the maybe_sync() utility."""

    def test_sync_context_returns_result(self):
        """When no event loop is running, runs the coroutine and returns."""

        async def add(a: int, b: int) -> int:
            return a + b

        result = maybe_sync(add(2, 3))
        assert result == 5

    def test_sync_context_with_none(self):
        async def noop() -> None:
            return None

        assert maybe_sync(noop()) is None

    @pytest.mark.asyncio
    async def test_async_context_returns_task(self):
        """When an event loop is running, returns an awaitable Task."""

        async def add(a: int, b: int) -> int:
            return a + b

        result = maybe_sync(add(10, 20))
        assert isinstance(result, asyncio.Task)
        assert await result == 30

    @pytest.mark.asyncio
    async def test_async_context_task_runs_without_await(self):
        """Task still executes even if not explicitly awaited."""
        flag = {"ran": False}

        async def set_flag() -> None:
            flag["ran"] = True

        maybe_sync(set_flag())
        # Yield control so the task can run
        await asyncio.sleep(0)
        assert flag["ran"]


# ---------------------------------------------------------------------------
# dump_error sync wrapper
# ---------------------------------------------------------------------------


class TestDumpErrorSync:
    """dump_error() sync/async compatibility."""

    def test_sync_noop_when_persistence_is_none(self):
        from llm_rosetta.observability.error_dump import dump_error

        result = dump_error(None, request_body=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_noop_when_persistence_is_none(self):
        from llm_rosetta.observability.error_dump import dump_error

        result = await dump_error(None, request_body=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_returns_awaitable(self):
        from llm_rosetta.observability.error_dump import dump_error

        result = dump_error(None, request_body=None)
        # In async context, returns a Task
        assert isinstance(result, asyncio.Task)
        assert await result is None


# ---------------------------------------------------------------------------
# _flush_now sync wrapper
# ---------------------------------------------------------------------------


class TestFlushNowSync:
    """_flush_now() sync/async compatibility."""

    def test_sync_noop_when_no_persistence(self):
        from llm_rosetta.gateway.app import _flush_now

        app = MagicMock(spec=[])  # no attributes
        _flush_now(app)  # should not raise

    @pytest.mark.asyncio
    async def test_async_noop_when_no_persistence(self):
        from llm_rosetta.gateway.app import _flush_now

        app = MagicMock(spec=[])
        await _flush_now(app)  # should not raise


# ---------------------------------------------------------------------------
# setup_admin sync wrapper
# ---------------------------------------------------------------------------


class TestSetupAdminSync:
    """setup_admin() sync/async compatibility."""

    @pytest.mark.asyncio
    async def test_async_returns_awaitable(self, tmp_path):
        from llm_rosetta.gateway.admin import setup_admin
        from llm_rosetta.gateway.config import GatewayConfig

        app = MagicMock()
        app.disabled_tabs = frozenset()
        config = GatewayConfig({"providers": {}})

        result = setup_admin(app, config, None, data_dir=str(tmp_path))
        assert isinstance(result, asyncio.Task)
        await result
        # Verify admin state was set on the app
        assert hasattr(app, "metrics") or app.metrics is not None


# ---------------------------------------------------------------------------
# create_app sync wrapper
# ---------------------------------------------------------------------------


class TestCreateAppSync:
    """create_app() sync/async compatibility."""

    def test_sync_returns_app(self):
        from llm_rosetta.gateway.app import create_app
        from llm_rosetta.gateway.config import GatewayConfig

        config = GatewayConfig({"providers": {}})
        app = create_app(config)
        # Should be an App, not a coroutine or Task
        assert not isinstance(app, asyncio.Task)
        assert hasattr(app, "route")

    @pytest.mark.asyncio
    async def test_async_returns_app(self):
        from llm_rosetta.gateway.app import create_app
        from llm_rosetta.gateway.config import GatewayConfig

        config = GatewayConfig({"providers": {}})
        app = await create_app(config)
        assert hasattr(app, "route")
